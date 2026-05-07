from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


AGENT_SHELL_MARKERS = [
    ("--dry_run", "scripts/llm.py does not expose --dry_run"),
    ('"agent"', "scripts/llm.py does not expose agent shell subcommand"),
    ("agent_test", "scripts/llm.py does not expose agent_test"),
    ("inspect", "scripts/llm.py does not support inspect"),
    ("trace", "scripts/llm.py does not support trace"),
    ("compare", "scripts/llm.py does not support compare"),
    ("case", "scripts/llm.py does not support case"),
    ("doctor", "scripts/llm.py does not support doctor"),
    ("cleanup", "scripts/llm.py does not support cleanup"),
    ("usage", "scripts/llm.py does not support usage"),
    ("max_api_calls", "scripts/llm.py missing API guard marker: max_api_calls"),
    ("max_tokens_total", "scripts/llm.py missing API guard marker: max_tokens_total"),
    ("timeout", "scripts/llm.py missing API guard marker: timeout"),
    ("max_turns", "scripts/llm.py missing API guard marker: max_turns"),
    ("max_same_prompt", "scripts/llm.py missing API guard marker: max_same_prompt"),
    ("safe_root", "scripts/llm.py missing cleanup guard marker: safe_root"),
    ("keep", "scripts/llm.py missing cleanup guard marker: keep"),
    ("dataset", "scripts/llm.py missing cleanup guard marker: dataset"),
    ("config", "scripts/llm.py missing cleanup guard marker: config"),
    ("models", "scripts/llm.py missing cleanup guard marker: models"),
    ("scripts", "scripts/llm.py missing cleanup guard marker: scripts"),
    ("release", "scripts/llm.py missing cleanup guard marker: release"),
    ("CONFIRM_TRAIN", "scripts/llm.py missing shell safety marker: CONFIRM_TRAIN"),
    ("pending_confirmation", "scripts/llm.py missing shell safety marker: pending_confirmation"),
    ("confirmed_train_candidate", "scripts/llm.py missing shell safety marker: confirmed_train_candidate"),
    ("benchmark_candidate", "scripts/llm.py missing shell safety marker: benchmark_candidate"),
    ("git push", "scripts/llm.py missing shell safety marker: git push"),
    (".env", "scripts/llm.py missing shell safety marker: .env"),
    ("/quiet", "scripts/llm.py missing shell safety marker: /quiet"),
    ("/verbose", "scripts/llm.py missing shell safety marker: /verbose"),
    ("concise_print", "scripts/llm.py missing shell safety marker: concise_print"),
    ("block_reason", "scripts/llm.py does not include dangerous-command blocking logic"),
    ("Blocked by Agent safety policy", "scripts/llm.py does not include dangerous-command blocking logic"),
    ("scripts/eval.py bench --config config/candidate.yaml", "scripts/llm.py does not keep benchmark execution blocked"),
    ("No safe benchmark command was found", "scripts/llm.py does not keep benchmark execution blocked"),
]


def check_agent_shell_behavior(warnings: list[str]) -> None:
    try:
        from scripts.llm import block_reason, parse_agent_command, safe_root, validate_cleanup_request

        behavior_checks = [
            (parse_agent_command("\u8fd0\u884c smoke")[0] == "safe_smoke", "parse smoke"),
            (parse_agent_command("\u8fd0\u884c audit")[0] == "safe_audit", "parse audit"),
            (parse_agent_command("\u68c0\u67e5 candidate")[0] == "inspect", "parse inspect"),
            (parse_agent_command("\u7efc\u5408\u8bca\u65ad")[0] == "doctor", "parse doctor"),
            (parse_agent_command("\u8bad\u7ec3 candidate")[0] == "train_candidate", "parse train"),
            (parse_agent_command("\u8dd1 benchmark")[0] == "benchmark_candidate", "parse benchmark"),
            (block_reason("git push origin main", "unknown") is not None, "block git push"),
            (block_reason("pip install x", "unknown") is not None, "block pip"),
            (block_reason("conda install x", "unknown") is not None, "block conda"),
            (block_reason("curl http://example.com", "unknown") is not None, "block curl"),
            (block_reason("wget http://example.com", "unknown") is not None, "block wget"),
            (block_reason("rm -rf outputs", "unknown") is not None, "block rm"),
            (block_reason("CUDA_VISIBLE_DEVICES=0 python main.py train", "unknown") is not None, "block cuda env"),
            (block_reason("report .env", "report") is not None, "block env read"),
            (block_reason("report README.md > .env", "report") is not None, "block redirect env"),
            (block_reason("运行 smoke >> outputs/log.txt", "safe_smoke") is not None, "block redirect log"),
            (safe_root(Path("outputs/llm_shell")) is True, "safe root llm_shell"),
            (safe_root(Path("outputs/llm")) is True, "safe root llm"),
            (safe_root(Path("outputs/llm_test")) is True, "safe root llm_test"),
            (safe_root(Path("outputs/llm_shell_test")) is True, "safe root llm_shell_test"),
            (safe_root(Path("outputs/llm_server_test")) is True, "safe root llm_server"),
            (safe_root(Path("models")) is False, "block root models"),
            (safe_root(Path("config")) is False, "block root config"),
            (safe_root(Path("dataset")) is False, "block root dataset"),
            (safe_root(Path("release")) is False, "block root release"),
            (safe_root(Path(".git")) is False, "block root git"),
        ]
        failed = [name for ok, name in behavior_checks if not ok]
        normalized = validate_cleanup_request(Path("outputs/llm_shell_test"), keep=-1)
        if normalized.get("normalized_keep") != 10:
            failed.append("cleanup keep normalization")
        if validate_cleanup_request(Path("models"), keep=10).get("status") != "blocked":
            failed.append("cleanup blocks models")
        if failed:
            warnings.append("Agent behavior checks failed: " + ", ".join(failed))
    except Exception as exc:
        warnings.append(f"Agent behavior checks failed to run: {exc}")


def check_agent_docs(readme_text: str, llm_script: Path, llm_text: str, analyzer_text: str, candidate_text: str, warnings: list[str]) -> None:
    readme_lower = readme_text.lower()
    if "LLM improves F1" in readme_text or "LLM improves Pair F1" in readme_text:
        warnings.append("README appears to claim LLM improves model F1")
    if "pair-prior" in readme_lower and "optional" not in readme_lower:
        warnings.append("README references pair-prior without optional/probe framing")
    if "read-only" not in readme_lower or "does not run training" not in readme_lower:
        warnings.append("README does not document Agent shell read-only safety")
    for cmd_name in ["inspect outputs/candidate", "trace config/candidate.yaml", "compare outputs/candidate", "case outputs/candidate", "doctor outputs/candidate"]:
        if cmd_name not in readme_text:
            warnings.append(f"README does not document Agent diagnostic command: {cmd_name}")
    for phrase in ["candidate training requires explicit confirmation", "unsafe benchmark", "max_api_calls", "max_tokens_total", "timeout", "repeated prompt", "keeps the most recent 10", "concise"]:
        if phrase not in readme_lower:
            warnings.append(f"README does not document Agent shell guard: {phrase}")
    if "agent" in candidate_text.lower() or "llm" in candidate_text.lower():
        warnings.append("config/candidate.yaml references agent or llm")
    if "LLM_API_KEY" not in analyzer_text or "LLM_MODEL" not in analyzer_text or "LLM_BASE_URL" not in analyzer_text:
        warnings.append("LLM analyzer does not read expected .env keys")
    if any("print(" in line and "api_key" in line for line in analyzer_text.splitlines()):
        warnings.append("LLM analyzer may print API key")

    forbidden_llm_exec = ["os.system", "Popen", "exec_command"]
    for item in forbidden_llm_exec:
        if item in llm_text:
            warnings.append(f"scripts/llm.py may execute forbidden workflow: {item}")
    forbidden_write = ["predictions.jsonl').write", 'predictions.jsonl").write', "benchmark.json').write", 'benchmark.json").write', "best.pt').write", 'best.pt").write']
    for item in forbidden_write:
        if item in llm_text:
            warnings.append(f"scripts/llm.py may write protected artifacts: {item}")

    if not llm_script.exists():
        warnings.append("scripts/llm.py is missing")
    for marker, msg in AGENT_SHELL_MARKERS:
        if marker not in llm_text:
            warnings.append(msg)
    if not all(command in llm_text for command in ["\\u68c0\\u67e5 candidate", "\\u7efc\\u5408\\u8bca\\u65ad", "\\u8fd0\\u884c smoke", "\\u8fd0\\u884c audit", "\\u8fdb\\u884c\\u8bad\\u7ec3"]):
        warnings.append("agent_test does not cover required chat-style commands")


def ensure_archive_paths(config: dict) -> None:
    data = config["data"]
    if Path(data["train_jsonl"]).exists():
        return
    raise SystemExit("ArchiveII split files are missing. Run `python scripts/data.py split --input dataset/processed/archivecheck.jsonl --out dataset/archive --mode random` first.")


def make_batch(config: dict, batches: int) -> tuple[dict, RNAOmniTokenizer]:
    from torch.utils.data import DataLoader
    from models.collator import RNAOmniCollator
    from models.dataset import RNAOmniDataset
    from models.token import RNAOmniTokenizer

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
    import torch
    from main import build_model, load_config, loss_from_batch, move_batch_to_device, resolve_device
    from models.omni import _pair_loss_mask
    from utils.struct import parse_dot_bracket

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
    import time
    import torch
    from main import build_model, load_config, move_batch_to_device, resolve_device

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
    check_agent_docs(readme_text, llm_script, llm_text, analyzer_text, candidate_text, warnings)
    check_agent_shell_behavior(warnings)

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

