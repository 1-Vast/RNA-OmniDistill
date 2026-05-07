from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.collator import RNAOmniCollator
from models.dataset import RNAOmniDataset
from models.token import RNAOmniTokenizer
from main import build_model, load_config, loss_from_batch, move_batch_to_device, resolve_device
from models.omni import _pair_loss_mask
from utils.struct import parse_dot_bracket


OLD_SCRIPT_FILES = [
    "scripts/run_potential_suite.py",
    "scripts/run_benchmark.py",
    "scripts/prepare_rna_dataset.py",
    "scripts/check_dataset.py",
    "scripts/make_splits.py",
    "scripts/run_realdata_smoke.py",
    "scripts/export_predictions.py",
    "scripts/analyze_training.py",
    "scripts/diagnose_predictions.py",
    "scripts/compare_benchmarks.py",
    "scripts/compare_core_results.py",
    "scripts/compare_ablations.py",
    "scripts/run_core_experiments.py",
    "scripts/run_ablations.py",
    "scripts/run_pairfix_sweep.py",
    "scripts/compare_pairfix_sweep.py",
    "scripts/summarize_model_potential.py",
    "scripts/evaluate_agent_potential.py",
    "scripts/debug_pair_alignment.py",
    "scripts/profile_runtime.py",
    "scripts/smoke_test_rna_omni.py",
]

OLD_MODEL_FILES = ["models/rna_omnidiffusion.py", "models/masking.py", "models/decoding.py"]
OLD_TOOL_FILES = [
    "data/__init__.py",
    "data/collator.py",
    "data/dataset.py",
    "data/token.py",
    "data/tokenizer.py",
    "utils/structure.py",
    "utils/metrics.py",
]
OLD_CONFIG_FILES = [
    "config/archive.yaml",
    "config/config.yaml",
    "config/config_archiveii.yaml",
    "config/config_rnastralign512.yaml",
    "config/config_external_archiveii.yaml",
    "config/relaxed.yaml",
    "config/pairfix.yaml",
    "config/strong.yaml",
    "config/ablate/hybrid.yaml",
    "config/ablate/nocond.yaml",
    "config/ablate/nomotif.yaml",
    "config/ablate/nopairmask.yaml",
    "config/ablate/pair.yaml",
    "config/ablate/token.yaml",
]

OLD_PATTERNS = [
    "run_potential_suite",
    "run_benchmark",
    "prepare_rna_dataset",
    "check_dataset",
    "make_splits",
    "run_realdata_smoke",
    "export_predictions",
    "analyze_training",
    "diagnose_predictions",
    "compare_benchmarks",
    "compare_core_results",
    "compare_ablations",
    "run_core_experiments",
    "run_ablations",
    "run_pairfix_sweep",
    "compare_pairfix_sweep",
    "summarize_model_potential",
    "evaluate_agent_potential",
    "rna_omnidiffusion",
    "data.tokenizer",
    "from data.",
    "import data.",
    "data/dataset.py",
    "data/collator.py",
    "data/token.py",
    "utils.structure",
    "utils.metrics",
    "models.masking",
    "models.decoding",
    "archiveii_full",
    "pairfix_sweep",
    "rna_omnidiffusion_v2",
    "processed_archiveii",
    "agent_potential",
    "model_potential",
]

KEEP_FILES = [
    "config/cpu.yaml: retained for CPU preflight and smoke checks",
]


def ensure_archive_paths(config: dict) -> None:
    data = config["data"]
    if Path(data["train_jsonl"]).exists():
        return
    raise SystemExit("ArchiveII split files are missing. Run `python scripts/data.py split --input dataset/processed/archivecheck.jsonl --out dataset/archive --mode random` first.")


def make_batch(config: dict, batches: int) -> tuple[dict, RNAOmniTokenizer]:
    ensure_archive_paths(config)
    dataset = RNAOmniDataset(config["data"]["train_jsonl"], max_length=int(config["data"]["max_length"]))
    samples = dataset.samples[: max(1, int(config["training"].get("batch_size", 8)) * batches)]
    tokenizer = RNAOmniTokenizer.from_samples(samples)
    collator = RNAOmniCollator(
        tokenizer,
        config["tasks"],
        pair_negative_ratio=int(config["training"].get("pair_negative_ratio", config["training"].get("pairRatio", 3))),
        seed=int(config["training"].get("seed", 42)),
        ablation=config.get("ablation", {}),
    )
    loader = DataLoader(samples, batch_size=int(config["training"].get("batch_size", 8)), collate_fn=collator)
    return next(iter(loader)), tokenizer


def run_align(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    batch, tokenizer = make_batch(config, args.batches)
    device = resolve_device(args.device)
    model = build_model(config, tokenizer, device)
    batch_device = move_batch_to_device(batch, device)
    outputs = model(
        input_ids=batch_device["input_ids"],
        attention_mask=batch_device["attention_mask"],
        segment_ids=batch_device["segment_ids"],
        task_ids=batch_device["task_ids"],
        time_steps=batch_device["time_steps"],
        seq_positions=batch_device["seq_positions"],
    )
    loss = loss_from_batch(outputs, batch_device, {
        "lambda_pair": float(config["training"].get("lambda_pair", config["training"].get("lambdaPair", 0.5))),
        "lambda_seq": float(config["training"].get("lambda_seq", 1.0)),
        "lambda_struct": float(config["training"].get("lambda_struct", 1.0)),
        "token_id_weights": None,
        "pair_pos_weight": config["training"].get("pair_positive_weight", config["training"].get("pairWeight", "auto")),
        "use_pair_loss": True,
        "pair_options": {
            "pairWeight": config["training"].get("pair_positive_weight", config["training"].get("pairWeight", "auto")),
            "pairRatio": int(config["training"].get("pair_negative_ratio", config["training"].get("pairRatio", 3))),
            "pairUpper": bool(config["training"].get("pairUpper", True)),
            "pairLoop": int(config["training"].get("pairLoop", 3)),
            "pairDiag": bool(config["training"].get("pairDiag", False)),
            "pairFloat": bool(config["training"].get("pairFloat", True)),
            "sampleNegOnGpu": bool(config["training"].get("sampleNegOnGpu", True)),
        },
    })
    pair_mask = _pair_loss_mask(
        batch_device["pair_mask"],
        batch_device["lengths"],
        {
            "pairUpper": bool(config["training"].get("pairUpper", True)),
            "pairLoop": int(config["training"].get("pairLoop", 3)),
            "pairDiag": bool(config["training"].get("pairDiag", False)),
        },
    ).detach().cpu()
    rows = []
    fail = []
    for i, seq in enumerate(batch["raw_seq"]):
        length = int(batch["lengths"][i])
        seq_pos = batch["seq_positions"][i, :length]
        struct_pos = batch["struct_positions"][i, :length]
        seq_ok = bool((batch["segment_ids"][i, seq_pos] == 1).all())
        struct_ok = bool((batch["segment_ids"][i, struct_pos] == 2).all())
        parsed = set(tuple(sorted(pair)) for pair in parse_dot_bracket(batch["raw_struct"][i]))
        labels = set()
        for a, b in torch.nonzero(batch["pair_labels"][i] > 0.5, as_tuple=False).tolist():
            labels.add((a, b))
        label_ok = parsed == labels
        lower_polluted = bool(torch.tril(pair_mask[i], diagonal=0).any())
        row = {
            "id": i,
            "length": length,
            "seqPositions": int((seq_pos >= 0).sum()),
            "structPositions": int((struct_pos >= 0).sum()),
            "seqSegmentOk": seq_ok,
            "structSegmentOk": struct_ok,
            "labelOk": label_ok,
            "lossMaskLowerPolluted": lower_polluted,
            "pos": float(loss.get("pos", torch.tensor(0.0))),
            "neg": float(loss.get("neg", torch.tensor(0.0))),
        }
        rows.append(row)
        if not seq_ok:
            fail.append("sequence positions enter non-sequence segment")
        if not struct_ok:
            fail.append("structure positions enter non-structure segment")
        if not label_ok:
            fail.append("pair labels differ from dot-bracket parse")
        if lower_polluted:
            fail.append("pair loss mask includes lower triangle or diagonal")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS" if not fail else "FAIL",
        "failures": sorted(set(fail)),
        "pos": float(loss.get("pos", torch.tensor(0.0))),
        "neg": float(loss.get("neg", torch.tensor(0.0))),
        "gap": float(loss.get("gap", torch.tensor(0.0))),
        "rankAcc": None if loss.get("rankAcc") is None else float(loss["rankAcc"]),
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "batch.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    lines = [
        "# Alignment Audit",
        "",
        f"Status: **{report['status']}**",
        f"Device: {report['device']} {report['gpu']}",
        f"Positive pairs: {report['pos']:.0f}",
        f"Negative pairs: {report['neg']:.0f}",
        f"Gap: {report['gap']:.4f}",
        f"RankAcc: {report['rankAcc']}",
        "",
        "## Checks",
        f"- position offset: {'FAIL' if any('positions' in item for item in fail) else 'PASS'}",
        f"- label offset: {'FAIL' if any('labels' in item for item in fail) else 'PASS'}",
        f"- mask pollution: {'FAIL' if any('mask' in item for item in fail) else 'PASS'}",
        f"- continue full training: {'yes' if not fail else 'no'}",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"align {report['status']} -> {out / 'report.md'}")


def run_profile(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    batch, tokenizer = make_batch(config, 1)
    device = resolve_device(args.device)
    model = build_model(config, tokenizer, device)
    batch = move_batch_to_device(batch, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    for _ in range(args.steps):
        start = time.time()
        opt.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            task_ids=batch["task_ids"],
            time_steps=batch["time_steps"],
            seq_positions=batch["seq_positions"],
        )
        forward = time.time()
        loss = torch.nn.functional.cross_entropy(outputs["token_logits"].view(-1, outputs["token_logits"].size(-1)), batch["labels"].view(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        end = time.time()
        times.append({"forward": forward - start, "step": end - start})
    avg_forward = sum(t["forward"] for t in times) / max(1, len(times))
    avg_step = sum(t["step"] for t in times) / max(1, len(times))
    report = {
        "device": device.type,
        "cuda": device.type == "cuda",
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "batch": int(batch["input_ids"].size(0)),
        "maxlen": int(config["data"]["max_length"]),
        "hidden": int(config["model"]["hidden_size"]),
        "layers": int(config["model"]["num_layers"]),
        "dataTime": 0.0,
        "forwardTime": avg_forward,
        "backwardTime": max(0.0, avg_step - avg_forward),
        "stepTime": avg_step,
        "decodeTime": 0.0,
        "epochTime": 0.0,
        "fullTime": avg_step * args.steps,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text("# Runtime Profile\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items()) + "\n", encoding="utf-8")
    print(f"profile -> {out / 'report.md'}")


def run_names(args: argparse.Namespace) -> None:
    roots = [Path("main.py"), Path("config"), Path("models"), Path("utils"), Path("scripts"), Path("README.md"), Path("INDEX.md")]
    bad = []
    for root in roots:
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if path.name == "__init__.py":
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            stem = path.stem
            if "_" in stem:
                bad.append(str(path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not bad else "FAIL"
    report = ["# Name Audit", "", f"Status: **{status}**", ""]
    if bad:
        report.extend(["## Files", *[f"- {item}" for item in bad]])
    else:
        report.append("No project-defined filenames with underscores were found.")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"status": status, "bad": bad}, indent=2) + "\n", encoding="utf-8")
    print(f"names {status} -> {out / 'report.md'}")
    if bad:
        raise SystemExit("Name audit failed.")


def iter_project_text_files() -> list[Path]:
    roots = [Path("main.py"), Path("config"), Path("models"), Path("utils"), Path("scripts"), Path("README.md"), Path("INDEX.md")]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json"}:
                continue
            files.append(path)
    return files


def run_clean(args: argparse.Namespace) -> None:
    missing_expected = OLD_SCRIPT_FILES + OLD_MODEL_FILES + OLD_TOOL_FILES + OLD_CONFIG_FILES
    remaining_files = [item for item in missing_expected if Path(item).exists()]
    pattern_hits: dict[str, list[str]] = {}
    for path in iter_project_text_files():
        if path == Path("scripts/audit.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in OLD_PATTERNS:
            if pattern in text:
                pattern_hits.setdefault(pattern, []).append(str(path))
    warnings = []
    if remaining_files:
        warnings.append("legacy files still exist")
    if pattern_hits:
        warnings.append("legacy references still exist")
    top_data = Path("data")
    if top_data.exists():
        warnings.append("top-level data Python package still exists")
    if not Path("dataset").exists():
        warnings.append("dataset data directory is missing")
    if not Path("scripts/data.py").exists():
        warnings.append("scripts/data.py is missing")
    if not Path("models/agent/__init__.py").exists():
        warnings.append("models/agent package is missing")

    llm_script = Path("scripts/llm.py")
    analyzer = Path("models/agent/analyzer.py")
    llm_text = llm_script.read_text(encoding="utf-8", errors="replace") if llm_script.exists() else ""
    analyzer_text = analyzer.read_text(encoding="utf-8", errors="replace") if analyzer.exists() else ""
    if not llm_script.exists():
        warnings.append("scripts/llm.py is missing")
    if "--dry_run" not in llm_text:
        warnings.append("scripts/llm.py does not expose --dry_run")
    if '"agent"' not in llm_text and "sub.add_parser(\"agent\"" not in llm_text:
        warnings.append("scripts/llm.py does not expose agent shell subcommand")
    if "agent_test" not in llm_text:
        warnings.append("scripts/llm.py does not expose agent_test")
    for command in ["inspect", "trace", "compare", "case", "doctor"]:
        if command not in llm_text:
            warnings.append(f"scripts/llm.py does not support {command}")
    if not all(command in llm_text for command in ["inspect outputs/candidate", "trace config/candidate.yaml", "compare outputs/candidate", "case outputs/candidate", "doctor outputs/candidate"]):
        warnings.append("agent_test does not cover all new diagnostic commands")
    if "safety_block_reason" not in llm_text or "Blocked by Agent safety policy" not in llm_text:
        warnings.append("scripts/llm.py does not include dangerous-command blocking logic")
    forbidden_llm_exec = ["subprocess", "os.system", "Popen", "exec_command", "scripts/eval.py bench"]
    for item in forbidden_llm_exec:
        if item in llm_text:
            warnings.append(f"scripts/llm.py may execute forbidden workflow: {item}")
    forbidden_write_markers = [
        "predictions.jsonl').write",
        'predictions.jsonl").write',
        "benchmark.json').write",
        'benchmark.json").write',
        "best.pt').write",
        'best.pt").write',
    ]
    for item in forbidden_write_markers:
        if item in llm_text:
            warnings.append(f"scripts/llm.py may write protected artifacts: {item}")
    if "LLM_API_KEY" not in analyzer_text or "LLM_MODEL" not in analyzer_text or "LLM_BASE_URL" not in analyzer_text:
        warnings.append("LLM analyzer does not read expected .env keys")
    if any("print(" in line and "api_key" in line for line in analyzer_text.splitlines()):
        warnings.append("LLM analyzer may print API key")

    def config_text(path: str) -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    candidate_text = config_text("config/candidate.yaml")
    fixed_text = config_text("config/fixed.yaml")
    best_text = config_text("release/best_config.yaml")
    for name, text in [
        ("config/candidate.yaml", candidate_text),
        ("config/fixed.yaml", fixed_text),
        ("release/best_config.yaml", best_text),
    ]:
        lowered = text.lower()
        if "semantic:" in lowered or "semantic.enabled: true" in lowered:
            warnings.append(f"{name} may enable semantic conditioning")
        if "constraint:" in lowered or "constraint.enabled: true" in lowered:
            warnings.append(f"{name} may enable constraint conditioning")
        if "pair_prior" in lowered or "pairprior" in lowered:
            warnings.append(f"{name} references pair-prior; candidate path should keep it disabled")

    readme_text = Path("README.md").read_text(encoding="utf-8", errors="replace") if Path("README.md").exists() else ""
    if "LLM improves F1" in readme_text or "LLM improves Pair F1" in readme_text:
        warnings.append("README appears to claim LLM improves model F1")
    if "pair-prior" in readme_text.lower() and "optional" not in readme_text.lower():
        warnings.append("README references pair-prior without optional/probe framing")
    if "read-only" not in readme_text.lower() or "does not run training" not in readme_text.lower():
        warnings.append("README does not document Agent shell read-only safety")
    for command in ["inspect outputs/candidate", "trace config/candidate.yaml", "compare outputs/candidate", "case outputs/candidate", "doctor outputs/candidate"]:
        if command not in readme_text:
            warnings.append(f"README does not document Agent diagnostic command: {command}")
    if "agent" in candidate_text.lower() or "llm" in candidate_text.lower():
        warnings.append("config/candidate.yaml references agent or llm")

    try:
        tracked = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8", errors="replace").splitlines()
    except Exception:
        tracked = []
    blocked_prefixes = ("outputs/", "dataset/raw/")
    blocked_exact = {".env"}
    blocked_suffixes = (".pt", ".pth", ".ckpt")
    tracked_blocked = [
        item for item in tracked
        if item in blocked_exact or item.startswith(blocked_prefixes) or item.endswith(blocked_suffixes)
    ]
    tracked_blocked.extend(
        item for item in tracked
        if item.startswith("dataset/processed/") and Path(item).exists() and Path(item).stat().st_size > 1_000_000
    )
    if tracked_blocked:
        warnings.append("blocked generated or secret files are tracked")
    status = "PASS" if not warnings else "FAIL"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "removed_files": [item for item in missing_expected if not Path(item).exists()],
        "remaining_legacy_files": remaining_files,
        "kept_legacy_files": KEEP_FILES,
        "remaining_references": pattern_hits,
        "top_level_data_package_exists": top_data.exists(),
        "dataset_dir_exists": Path("dataset").exists(),
        "scripts_data_cli_exists": Path("scripts/data.py").exists(),
        "agent_files": sorted(str(path) for path in Path("models/agent").rglob("*.py")) if Path("models/agent").exists() else [],
        "tracked_blocked_files": tracked_blocked,
        "warnings": warnings,
        "recommended_next_command": "conda run -n DL python scripts\\run.py ablate --config config/fixed.yaml --only full nopair nonuss random --device cuda --decode nussinov --bench_workers 8 --bench_profile --bench_resume",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Cleanup Audit",
        "",
        f"Status: **{status}**",
        "",
        "## Removed Files",
        *[f"- {item}" for item in report["removed_files"]],
        "",
        "## Kept Legacy Files With Reason",
        *[f"- {item}" for item in KEEP_FILES],
        "",
        "## Remaining Warnings",
    ]
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- none")
    if remaining_files:
        lines += ["", "## Remaining Legacy Files", *[f"- {item}" for item in remaining_files]]
    if pattern_hits:
        lines += ["", "## Remaining Legacy References"]
        for pattern, paths in sorted(pattern_hits.items()):
            lines.append(f"- `{pattern}`: {', '.join(sorted(set(paths)))}")
    lines += ["", "## Recommended Next Command", "", f"`{report['recommended_next_command']}`"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"clean {status} -> {out / 'report.md'}")
    if status != "PASS":
        raise SystemExit("Cleanup audit failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RNA-OmniDiffusion runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    align = sub.add_parser("align")
    align.add_argument("--config", default="config/fixed.yaml")
    align.add_argument("--batches", type=int, default=1)
    align.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    align.add_argument("--out", default="outputs/align")
    align.set_defaults(func=run_align)
    profile = sub.add_parser("profile")
    profile.add_argument("--config", default="config/fixed.yaml")
    profile.add_argument("--steps", type=int, default=2)
    profile.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    profile.add_argument("--out", default="outputs/profile")
    profile.set_defaults(func=run_profile)
    names = sub.add_parser("names")
    names.add_argument("--out", default="outputs/name")
    names.set_defaults(func=run_names)
    clean = sub.add_parser("clean")
    clean.add_argument("--out", default="outputs/clean")
    clean.set_defaults(func=run_clean)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

