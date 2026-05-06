from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


VARIANTS = ["orig", "relax", "fix", "fixed"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        return proc.wait()


def output_dir_from_config(config_path: Path) -> Path:
    config = load_yaml(config_path)
    training = config.get("training", {})
    return Path(training.get("out") or training.get("output_dir") or "outputs/archive")


def quick_potential_config(config_path: Path, out: Path) -> Path:
    config = load_yaml(config_path)
    config.setdefault("training", {})
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = min(int(config["training"].get("batch_size", 8)), 2)
    config["training"]["warmup_steps"] = 1
    config["training"]["log_every"] = 1
    config["training"]["out"] = str(out)
    config["training"]["output_dir"] = str(out)
    config.setdefault("data", {})
    config["data"]["max_length"] = min(int(config["data"].get("max_length", 512)), 128)
    config.setdefault("model", {})
    config["model"]["hidden_size"] = 64
    config["model"]["num_layers"] = 2
    config["model"]["num_heads"] = 4
    config["model"]["max_position_embeddings"] = 512
    if str(config["model"].get("pairhead", "")).lower() == "pairmlp":
        config["model"]["pairhidden"] = 64
    config.setdefault("decoding", {})
    config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
    path = out / "config.yaml"
    write_yaml(path, config)
    return path


def require_data(config_path: Path) -> None:
    config = load_yaml(config_path)
    data = config.get("data", {})
    missing = []
    for key in ("train_jsonl", "val_jsonl", "test_jsonl"):
        value = data.get(key)
        if not value or not Path(value).exists():
            missing.append(f"{key}={value}")
    if missing:
        raise SystemExit(
            "ArchiveII JSONL files are missing; full mode cannot use toy fallback. "
            "Prepare data first with scripts/data.py fetch/prep/check/split. Missing: "
            + ", ".join(missing)
        )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def last_train_row(out: Path) -> dict:
    rows = read_jsonl(out / "trainlog.jsonl")
    return rows[-1] if rows else {}


def classify_failure(step: str, log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower() if log_path.exists() else ""
    if "out of memory" in text or "cuda error" in text:
        return "显存问题"
    if step in {"data", "data_check"}:
        return "数据问题"
    if step == "train":
        return "训练问题"
    if step == "benchmark":
        return "benchmark 问题"
    return "流程问题"


def build_full_summary(
    out: Path,
    config_path: Path,
    elapsed: float,
    completed: bool,
    failed_step: str | None = None,
    failed_command: str | None = None,
    log_path: Path | None = None,
) -> dict:
    benchmark = read_json(out / "benchmark.json")
    analysis = read_json(out / "analysis.json")
    diagnosis = read_json(out / "diagnosis.json")
    train_row = last_train_row(out)
    best = analysis.get("best") or train_row
    model = benchmark.get("overall", {}).get("model", {})
    random_base = benchmark.get("overall", {}).get("random", {})
    all_base = benchmark.get("overall", {}).get("all", {})
    avg_pred = as_float(model.get("avg_pred_pair_count"))
    avg_true = as_float(model.get("avg_true_pair_count"))
    pair_count_ratio = avg_pred / max(1e-8, avg_true)
    gap = as_float(best.get("gap", best.get("pair_logit_gap")))
    rank = best.get("rankAcc", best.get("pair_ranking_accuracy_sampled"))
    rank_float = None if rank is None else as_float(rank)
    pair_f1 = as_float(model.get("pair_f1"))
    random_f1 = as_float(random_base.get("pair_f1"))
    all_dot = as_float(model.get("all_dot_ratio"))
    valid = as_float(model.get("valid_structure_rate"))
    over_pairing = pair_count_ratio > 1.5
    severe_over_pairing = pair_count_ratio > 2.0
    all_dot_collapse = all_dot > 0.3
    ranking_failure = gap <= 0.0 or (rank_float is not None and rank_float <= 0.5)
    decoding_driven = pair_f1 > random_f1 and ranking_failure
    passed = (
        completed
        and pair_f1 > random_f1
        and all_dot < 0.3
        and valid >= 0.95
        and 0.5 <= pair_count_ratio <= 1.5
        and gap > 0.0
        and rank_float is not None
        and rank_float > 0.5
    )
    if not completed:
        decision = "Decision: Full run failed. Do not run core ablation."
        recommended_config = None
    elif passed:
        decision = "Decision: Run core ablation next."
        recommended_config = None
    elif severe_over_pairing:
        decision = "Decision: Severe over-pairing. Do not run ablation. Try config/strict.yaml."
        recommended_config = "config/strict.yaml"
    elif over_pairing:
        decision = "Decision: Mild over-pairing. Do not run ablation yet. Try config/mild.yaml."
        recommended_config = "config/mild.yaml"
    elif ranking_failure:
        decision = "Decision: Pair head ranking is unstable. Do not tune decoding. Try config/stable.yaml or inspect pair labels."
        recommended_config = "config/stable.yaml"
    elif all_dot_collapse:
        decision = "Decision: All-dot risk. Do not run ablation. Inspect structure loss and decoding."
        recommended_config = None
    else:
        decision = "Decision: Full metrics are mixed. Do not run ablation until pair head diagnostics are reviewed."
        recommended_config = None
    summary = {
        "completed": completed,
        "config": str(config_path),
        "output_dir": str(out),
        "decode_method": benchmark.get("decode_method", "unknown"),
        "decode_warning": benchmark.get("decode_warning", ""),
        "elapsed_seconds": elapsed,
        "device": train_row.get("device", "unknown"),
        "cuda": train_row.get("cuda"),
        "gpu": train_row.get("gpu", "unknown"),
        "best_epoch": best.get("epoch"),
        "test_pair_precision": as_float(model.get("pair_precision")),
        "test_pair_recall": as_float(model.get("pair_recall")),
        "test_pair_f1": pair_f1,
        "random_valid_pair_baseline_f1": random_f1,
        "all_dot_baseline_f1": as_float(all_base.get("pair_f1")),
        "all_dot_ratio": all_dot,
        "valid_structure_rate": valid,
        "canonical_pair_ratio": as_float(model.get("canonical_pair_ratio")),
        "avg_pred_pair_count": avg_pred,
        "avg_true_pair_count": avg_true,
        "pair_count_ratio": pair_count_ratio,
        "positive_pair_logit_mean": as_float(best.get("posLogit", best.get("positive_pair_logit_mean"))),
        "negative_pair_logit_mean": as_float(best.get("negLogit", best.get("negative_pair_logit_mean"))),
        "pair_logit_gap": gap,
        "rankAcc": rank_float,
        "over_pairing": over_pairing,
        "severe_over_pairing": severe_over_pairing,
        "all_dot_collapse": all_dot_collapse,
        "pair_head_ranking_failure": ranking_failure,
        "likely_decoding_driven_improvement": decoding_driven,
        "recommend_core_ablation": passed,
        "recommended_config": recommended_config,
        "decision": decision,
        "diagnosis_count": diagnosis.get("count"),
    }
    if failed_step:
        summary["failed_step"] = failed_step
    if failed_command:
        summary["failed_command"] = failed_command
    if log_path:
        summary["log_path"] = str(log_path)
        summary["failure_type"] = classify_failure(failed_step or "", log_path)
    return summary


def write_full_report(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "full.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    elapsed_min = as_float(summary.get("elapsed_seconds")) / 60.0
    rank_text = "" if summary.get("rankAcc") is None else f"{summary['rankAcc']:.4f}"
    lines = [
        "# ArchiveII Full Decision",
        "",
        f"- Full completed: {summary['completed']}",
        f"- Config: `{summary.get('config')}`",
        f"- Decode method: {summary.get('decode_method')}",
        f"- Device: {summary.get('device')} CUDA={summary.get('cuda')} GPU={summary.get('gpu')}",
        f"- Total time: {elapsed_min:.2f} min",
        f"- Best epoch: {summary.get('best_epoch')}",
        "",
        "## Metrics",
        "",
        *(
            [
                "Benchmark metrics are unavailable because full training did not complete; numeric benchmark fields in `full.json` are placeholders.",
                "",
            ]
            if not summary.get("completed")
            else []
        ),
        f"- test pair precision: {summary['test_pair_precision']:.4f}",
        f"- test pair recall: {summary['test_pair_recall']:.4f}",
        f"- test pair F1: {summary['test_pair_f1']:.4f}",
        f"- random valid-pair baseline F1: {summary['random_valid_pair_baseline_f1']:.4f}",
        f"- all-dot baseline F1: {summary['all_dot_baseline_f1']:.4f}",
        f"- all-dot ratio: {summary['all_dot_ratio']:.4f}",
        f"- valid structure rate: {summary['valid_structure_rate']:.4f}",
        f"- canonical pair ratio: {summary['canonical_pair_ratio']:.4f}",
        f"- avg pred pair count: {summary['avg_pred_pair_count']:.4f}",
        f"- avg true pair count: {summary['avg_true_pair_count']:.4f}",
        f"- pair_count_ratio: {summary['pair_count_ratio']:.4f}",
        f"- positive_pair_logit_mean: {summary['positive_pair_logit_mean']:.4f}",
        f"- negative_pair_logit_mean: {summary['negative_pair_logit_mean']:.4f}",
        f"- pair_logit_gap: {summary['pair_logit_gap']:.4f}",
        f"- rankAcc: {rank_text}",
        "",
        "## Risk Checks",
        "",
        f"- over-pairing: {summary['over_pairing']}",
        f"- severe over-pairing: {summary['severe_over_pairing']}",
        f"- all-dot collapse: {summary['all_dot_collapse']}",
        f"- pair head ranking failure: {summary['pair_head_ranking_failure']}",
        f"- likely decoding-driven improvement: {summary['likely_decoding_driven_improvement']}",
        "",
        "## Decision",
        "",
        summary["decision"],
    ]
    if summary.get("recommend_core_ablation"):
        lines += [
            "",
            "Next command:",
            "",
            "```powershell",
            "conda run -n DL python scripts\\run.py ablate --config config/fixed.yaml --only full nopair nonuss random --device cuda",
            "```",
        ]
    if summary.get("recommended_config"):
        lines += ["", f"Recommended intervention config: `{summary['recommended_config']}`"]
    if summary.get("decode_method") == "greedy":
        lines += [
            "",
            "## Decode Warning",
            "",
            "Greedy decode is a fast approximate benchmark path. It is used to test pair-head structural signal and benchmark throughput. Strict Nussinov benchmark should still be run for final reporting if feasible.",
            "",
            str(summary.get("decode_warning") or ""),
        ]
    if summary.get("failed_step"):
        lines += [
            "",
            "## Failure",
            "",
            f"- failed_step: {summary.get('failed_step')}",
            f"- failed_command: `{summary.get('failed_command')}`",
            f"- log_path: `{summary.get('log_path')}`",
            f"- failure_type: {summary.get('failure_type')}",
        ]
        if summary.get("failure_note"):
            lines.append(f"- failure_note: {summary.get('failure_note')}")
    (out / "full.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def quick_config(name: str, mode: str) -> Path:
    config = load_yaml(Path("config") / f"{name}.yaml")
    if mode == "quick":
        config.setdefault("training", {})
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = min(int(config["training"].get("batch_size", 8)), 2)
        config["training"]["warmup_steps"] = 1
        config["training"]["log_every"] = 1
        config["training"]["out"] = f"outputs/sweep/{name}"
        config["training"]["output_dir"] = f"outputs/sweep/{name}"
        config.setdefault("data", {})
        config["data"]["max_length"] = min(int(config["data"].get("max_length", 512)), 128)
        config.setdefault("model", {})
        config["model"]["hidden_size"] = 64
        config["model"]["num_layers"] = 2
        config["model"]["num_heads"] = 4
        config["model"]["max_position_embeddings"] = 512
        config.setdefault("decoding", {})
        config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
    else:
        config["training"]["out"] = f"outputs/sweep/{name}"
        config["training"]["output_dir"] = f"outputs/sweep/{name}"
    path = Path("outputs/sweep") / name / "config.yaml"
    write_yaml(path, config)
    return path


def read_metric(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("overall", {}).get("model", {})


def read_analysis(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(variant: str, out: Path) -> dict:
    benchmark = read_json(out / "benchmark.json")
    if not benchmark:
        raise SystemExit(f"Missing benchmark.json for variant {variant}: {out / 'benchmark.json'}")
    metrics = benchmark.get("overall", {}).get("model", {})
    avg_true = as_float(metrics.get("avg_true_pair_count"))
    pair_ratio = as_float(metrics.get("avg_pred_pair_count")) / max(1e-8, avg_true)
    return {
        "variant": variant,
        "decode_method": benchmark.get("decode_method", "unknown"),
        "pair_head_available": benchmark.get("pair_head_available"),
        "pair_f1": as_float(metrics.get("pair_f1")),
        "pair_precision": as_float(metrics.get("pair_precision")),
        "pair_recall": as_float(metrics.get("pair_recall")),
        "valid_structure_rate": as_float(metrics.get("valid_structure_rate")),
        "all_dot_ratio": as_float(metrics.get("all_dot_ratio")),
        "pair_count_ratio": pair_ratio,
        "benchmark_seconds": as_float(benchmark.get("benchmark_seconds")),
    }


def write_ablate_summary(rows: list[dict], root: Path, quick: bool) -> None:
    prefix = "quick_" if quick else ""
    json_path = root / f"{prefix}summary.json"
    csv_path = root / f"{prefix}summary.csv"
    md_path = root / f"{prefix}summary.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Variant | Decode Method | Pair F1 | Precision | Recall | Valid | All-dot | Pair Ratio | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['decode_method']} | {row['pair_f1']:.4f} | "
            f"{row['pair_precision']:.4f} | {row['pair_recall']:.4f} | {row['valid_structure_rate']:.4f} | "
            f"{row['all_dot_ratio']:.4f} | {row['pair_count_ratio']:.4f} | {row['benchmark_seconds']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablate_decision(rows: list[dict], root: Path) -> None:
    by_name = {row["variant"]: row for row in rows}
    full = by_name["full"]
    delta_nopair = full["pair_f1"] - by_name["nopair"]["pair_f1"]
    delta_nonuss = full["pair_f1"] - by_name["nonuss"]["pair_f1"]
    delta_random = full["pair_f1"] - by_name["random"]["pair_f1"]
    over = "none"
    if full["pair_count_ratio"] > 2.0:
        over = "severe over-pairing"
    elif full["pair_count_ratio"] > 1.5:
        over = "mild over-pairing"
    fallback_explicit = by_name["nopair"]["decode_method"] == "tokenfallback" and by_name["nonuss"]["decode_method"] in {"token", "greedyfallback"}
    fallback_valid = (
        fallback_explicit
        and by_name["nopair"]["valid_structure_rate"] >= 0.95
        and by_name["nonuss"]["valid_structure_rate"] >= 0.95
    )
    paper_ready = fallback_valid and over == "none"
    if fallback_valid:
        fallback_text = "strictly comparable"
    elif fallback_explicit:
        fallback_text = "explicit but invalid; use nopair/nonuss only as failure-mode diagnostics"
    else:
        fallback_text = "not comparable"
    lines = [
        "# Core Ablation Decision",
        "",
        f"- delta_nopair: {delta_nopair:.4f}",
        f"- delta_nonuss: {delta_nonuss:.4f}",
        f"- delta_random: {delta_random:.4f}",
        "",
        f"- Pair head contribution: {'clear' if delta_nopair > 0.03 else 'weak'}",
        f"- Nussinov contribution: {'clear' if delta_nonuss > 0.10 or by_name['nonuss']['valid_structure_rate'] < full['valid_structure_rate'] - 0.05 else 'weak'}",
        f"- Masking contribution: {'clear' if delta_random > 0.02 else 'weak'}",
        f"- Over-pairing: {over} (full pair_count_ratio={full['pair_count_ratio']:.4f})",
        f"- Fallback validity: {fallback_text}",
        "",
        "## Notes",
        "",
        "- `nopair` uses tokenfallback and has no pair head; it does not create dummy pair logits.",
        "- `nonuss` removes the strict DP constraint and is not a final strict structure metric.",
        "- If `nonuss` falls back to greedyfallback, treat it only as a constraint-removal failure mode.",
        "",
        f"Final recommendation: {'can enter the paper ablation table with fallback caveats' if paper_ready else 'do not use directly as a strict paper table without another diagnostic pass'}",
    ]
    if over == "mild over-pairing":
        lines.append("Recommended next config if retuning is needed: `config/mild.yaml`.")
    elif over == "severe over-pairing":
        lines.append("Recommended next config if retuning is needed: `config/strict.yaml`.")
    elif not fallback_valid:
        lines.append("Recommended next config is not a training config; inspect token fallback validity first.")
    (root / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> None:
    Path("outputs/sweep").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in VARIANTS:
        config_path = quick_config(name, args.mode)
        out = Path("outputs/sweep") / name
        train = [sys.executable, "main.py", "train", "--config", str(config_path), "--device", args.device]
        if args.mode == "quick":
            train += ["--train_subset", "32", "--max_steps", "8"]
        run_cmd(train)
        run_cmd([sys.executable, "scripts/eval.py", "bench", "--config", str(config_path), "--ckpt", str(out / "best.pt"), "--split", "test", "--out", str(out / "benchmark.json"), "--device", args.device, "--samples", "128" if args.mode == "quick" else "0"])
        run_cmd([sys.executable, "scripts/eval.py", "analyze", "--log", str(out / "trainlog.jsonl"), "--out", str(out / "analysis.json")])
        run_cmd([sys.executable, "scripts/eval.py", "diagnose", "--pred", str(out / "predictions.jsonl"), "--out", str(out / "diagnosis.json")])
        metrics = read_metric(out / "benchmark.json")
        analysis = read_analysis(out / "analysis.json")
        base = 0.0
        if name != "archive":
            base = rows[0].get("pair_f1", 0.0) if rows else 0.0
        ratio = float(metrics.get("avg_pred_pair_count", 0.0)) / max(1e-8, float(metrics.get("avg_true_pair_count", 0.0)))
        best = analysis.get("best", {})
        rows.append({
            "variant": name,
            "pair_f1": float(metrics.get("pair_f1", 0.0)),
            "base_f1": base,
            "gap": float(best.get("gap", analysis.get("gap", 0.0)) or 0.0),
            "rankAcc": best.get("rankAcc", analysis.get("rankAcc")),
            "all_dot": float(metrics.get("all_dot_ratio", 0.0)),
            "valid": float(metrics.get("valid_structure_rate", 0.0)),
            "pair_ratio": ratio,
        })
    out = Path("outputs/sweep")
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Pair F1 | Base F1 | Gap | Rank Acc | All-dot | Valid | Pair Ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        rank = row["rankAcc"]
        rank_text = "" if rank is None else f"{float(rank):.4f}"
        lines.append(f"| {row['variant']} | {row['pair_f1']:.4f} | {row['base_f1']:.4f} | {row['gap']:.4f} | {rank_text} | {row['all_dot']:.4f} | {row['valid']:.4f} | {row['pair_ratio']:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"sweep -> {out / 'summary.md'}")


def run_potential(args: argparse.Namespace) -> None:
    source_config = Path(args.config) if args.config else Path("config/fixed.yaml" if args.set == "archive" else f"config/{args.set}.yaml")
    if not source_config.exists():
        raise SystemExit(f"Config does not exist: {source_config}")
    require_data(source_config)
    base_out = output_dir_from_config(source_config)
    out = Path(str(base_out) + "_quick") if args.mode == "quick" and args.config else base_out
    config_path = quick_potential_config(source_config, out) if args.mode == "quick" else source_config
    if args.mode == "full":
        out = output_dir_from_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    start = time.time()
    steps = [
        ("train", [sys.executable, "main.py", "train", "--config", str(config_path), "--device", args.device]),
        ("benchmark", [sys.executable, "scripts/eval.py", "bench", "--config", str(config_path), "--ckpt", str(out / "best.pt"), "--split", "test", "--out", str(out / "benchmark.json"), "--device", args.device, "--decode", args.decode]),
        ("analyze", [sys.executable, "scripts/eval.py", "analyze", "--log", str(out / "trainlog.jsonl"), "--out", str(out / "analysis.json")]),
        ("diagnose", [sys.executable, "scripts/eval.py", "diagnose", "--pred", str(out / "predictions.jsonl"), "--out", str(out / "diagnosis.json")]),
    ]
    if args.mode == "quick":
        steps[0][1].extend(["--train_subset", "32", "--max_steps", "8"])
        steps[1][1].extend(["--samples", "128"])
    if args.bench_limit:
        steps[1][1].extend(["--limit", str(args.bench_limit)])
    if args.bench_profile:
        steps[1][1].append("--profile")
    if args.bench_resume:
        steps[1][1].append("--resume")
    if args.bench_batch:
        steps[1][1].extend(["--batch", str(args.bench_batch)])
    if args.stage_logits:
        steps[1][1].append("--stage_logits")
    if args.decode_only:
        steps[1][1].append("--decode_only")
    if args.logits_file:
        steps[1][1].extend(["--logits_file", args.logits_file])
    if args.bench_workers is not None:
        steps[1][1].extend(["--workers", str(args.bench_workers)])
    if args.bench_chunksize is not None:
        steps[1][1].extend(["--chunksize", str(args.bench_chunksize)])
    if args.threshold is not None:
        steps[1][1].extend(["--threshold", str(args.threshold)])
    if args.gamma is not None:
        steps[1][1].extend(["--gamma", str(args.gamma)])
    if args.source:
        steps[1][1].extend(["--source", args.source])
    if args.token_alpha is not None:
        steps[1][1].extend(["--token_alpha", str(args.token_alpha)])
    if args.scan:
        steps[1][1].extend(["--scan", args.scan])
    for name, cmd in steps:
        log_path = logs / f"{name}.log"
        code = run_logged(cmd, log_path)
        if code != 0:
            summary = build_full_summary(
                out,
                config_path,
                time.time() - start,
                completed=False,
                failed_step=name,
                failed_command=" ".join(cmd),
                log_path=log_path,
            )
            write_full_report(summary, out)
            raise SystemExit(f"{name} failed; see {log_path} and {out / 'full.md'}")
    summary = build_full_summary(out, config_path, time.time() - start, completed=True)
    write_full_report(summary, out)
    print(f"full report -> {out / 'full.md'}")


def run_ablate(args: argparse.Namespace) -> None:
    root = Path("outputs/ablate")
    root.mkdir(parents=True, exist_ok=True)
    names = args.only or ["full", "nonuss", "random"]
    for name in names:
        base = load_yaml(Path(args.config))
        patch = load_yaml(Path("config/ablate") / f"{name}.yaml")
        base.setdefault("training", {}).update(patch.get("training", {}))
        if "output_dir" in base["training"]:
            base["training"]["out"] = base["training"]["output_dir"]
        base.setdefault("ablation", {}).update(patch.get("ablation", {}))
        base.setdefault("decoding", {}).update(patch.get("decoding", {}))
        path = Path("outputs/ablate") / name / "config.yaml"
        write_yaml(path, base)
        out = output_dir_from_config(path)
        out.mkdir(parents=True, exist_ok=True)
        train_cmd = [sys.executable, "main.py", "train", "--config", str(path), "--device", args.device]
        if args.quick:
            train_cmd.extend(["--train_subset", "32", "--max_steps", "8"])
        decode = args.decode
        if name == "nopair":
            decode = "tokenfallback"
        elif name == "nonuss":
            decode = "token"
        bench_cmd = [
            sys.executable,
            "scripts/eval.py",
            "bench",
            "--config",
            str(path),
            "--ckpt",
            str(out / "best.pt"),
            "--split",
            "test",
            "--out",
            str(out / "benchmark.json"),
            "--device",
            args.device,
            "--decode",
            decode,
        ]
        if args.quick:
            bench_cmd.extend(["--limit", "32"])
        if args.bench_workers is not None:
            bench_cmd.extend(["--workers", str(args.bench_workers)])
        if args.bench_profile:
            bench_cmd.append("--profile")
        if args.bench_resume:
            bench_cmd.append("--resume")
        if name in {"full", "random"} and decode == "nussinov":
            bench_cmd.append("--stage_logits")
        if args.dry_run:
            print(" ".join(train_cmd))
            print(" ".join(bench_cmd))
            print(" ".join([sys.executable, "scripts/eval.py", "analyze", "--log", str(out / "trainlog.jsonl"), "--out", str(out / "analysis.json")]))
            print(" ".join([sys.executable, "scripts/eval.py", "diagnose", "--pred", str(out / "predictions.jsonl"), "--out", str(out / "diagnosis.json")]))
            continue
        reuse_full = name == "full" and not args.quick and Path("outputs/fixed/best.pt").exists()
        if reuse_full:
            (out / "reuse_note.json").write_text(
                json.dumps(
                    {
                        "source_ckpt": "outputs/fixed/best.pt",
                        "source_benchmark": "outputs/fixed/benchmark.json",
                        "full_retrained": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            bench_cmd[bench_cmd.index("--ckpt") + 1] = "outputs/fixed/best.pt"
            trainlog = Path("outputs/fixed/trainlog.jsonl")
            if trainlog.exists():
                shutil.copyfile(trainlog, out / "trainlog.jsonl")
        else:
            run_cmd(train_cmd)
        run_cmd(bench_cmd)
        run_cmd([sys.executable, "scripts/eval.py", "analyze", "--log", str(out / "trainlog.jsonl"), "--out", str(out / "analysis.json")])
        run_cmd([sys.executable, "scripts/eval.py", "diagnose", "--pred", str(out / "predictions.jsonl"), "--out", str(out / "diagnosis.json")])
        benchmark = read_json(out / "benchmark.json")
        if "decode_method" not in benchmark:
            raise SystemExit(f"benchmark.json for {name} is missing decode_method")
    if args.dry_run:
        return
    rows = [metric_row(name, output_dir_from_config(Path("outputs/ablate") / name / "config.yaml")) for name in names]
    write_ablate_summary(rows, root, args.quick)
    if not args.quick:
        required = {"full", "nopair", "nonuss", "random"}
        if required.issubset(set(names)):
            write_ablate_decision(rows, root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact RNA-OmniDiffusion workflows.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--mode", choices=["quick", "full"], default="quick")
    sweep.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    sweep.set_defaults(func=run_sweep)
    potential = sub.add_parser("potential")
    potential.add_argument("--set", default="archive")
    potential.add_argument("--config")
    potential.add_argument("--mode", choices=["quick", "full"], default="quick")
    potential.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    potential.add_argument("--decode", choices=["nussinov", "greedy"], default="nussinov")
    potential.add_argument("--bench_limit", type=int)
    potential.add_argument("--bench_profile", action="store_true")
    potential.add_argument("--bench_resume", action="store_true")
    potential.add_argument("--bench_batch", type=int)
    potential.add_argument("--stage_logits", action="store_true")
    potential.add_argument("--decode_only", action="store_true")
    potential.add_argument("--logits_file")
    potential.add_argument("--bench_workers", type=int)
    potential.add_argument("--bench_chunksize", type=int)
    potential.add_argument("--threshold", type=float)
    potential.add_argument("--gamma", type=float)
    potential.add_argument("--source", choices=["pair", "hybrid"])
    potential.add_argument("--token_alpha", type=float)
    potential.add_argument("--scan")
    potential.set_defaults(func=run_potential)
    ablate = sub.add_parser("ablate")
    ablate.add_argument("--config", default="config/fixed.yaml")
    ablate.add_argument("--only", nargs="*")
    ablate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ablate.add_argument("--decode", choices=["nussinov", "greedy"], default="nussinov")
    ablate.add_argument("--bench_workers", type=int)
    ablate.add_argument("--bench_profile", action="store_true")
    ablate.add_argument("--bench_resume", action="store_true")
    ablate.add_argument("--dry_run", action="store_true")
    ablate.add_argument("--quick", action="store_true")
    ablate.set_defaults(func=run_ablate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
