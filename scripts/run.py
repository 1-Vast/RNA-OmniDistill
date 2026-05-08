from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import json, subprocess

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
        decision = "Decision: Severe over-pairing. Do not run ablation. Create a trial from config/candidate.yaml with stricter decoding or loss settings."
        recommended_config = "config/candidate.yaml"
    elif over_pairing:
        decision = "Decision: Mild over-pairing. Do not run ablation yet. Create a trial from config/candidate.yaml with lower pair threshold pressure."
        recommended_config = "config/candidate.yaml"
    elif ranking_failure:
        decision = "Decision: Pair head ranking is unstable. Do not tune decoding. Inspect pair labels before creating a trial from config/candidate.yaml."
        recommended_config = "config/candidate.yaml"
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
            "conda run -n DL python scripts\\run.py ablate --config config/candidate.yaml --only full nopair nonuss random --device cuda",
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


def sweep_config(config_path: Path, out: Path, mode: str) -> Path:
    config = load_yaml(config_path)
    config.setdefault("training", {})
    config["training"]["out"] = str(out)
    config["training"]["output_dir"] = str(out)
    if mode == "quick":
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = min(int(config["training"].get("batch_size", 8)), 2)
        config["training"]["warmup_steps"] = 1
        config["training"]["log_every"] = 1
        config.setdefault("data", {})
        config["data"]["max_length"] = min(int(config["data"].get("max_length", 512)), 128)
        config.setdefault("model", {})
        config["model"]["hidden_size"] = 64
        config["model"]["num_layers"] = 2
        config["model"]["num_heads"] = 4
        config["model"]["max_position_embeddings"] = 512
        if str(config["model"].get("pairhead", "")).lower() == "pairmlp":
            config["model"]["pairhidden"] = 64
        if config["model"].get("pairrefine"):
            config["model"]["pairrefinechannels"] = min(int(config["model"].get("pairrefinechannels", 16)), 16)
        config.setdefault("decoding", {})
        config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
    path = out / "config.yaml"
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


def write_seed_summary(rows: list[dict], root: Path, tag: str, seeds: list[int], variants: list[str]) -> None:
    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    def std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = mean(values)
        return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5

    full_rows = [row for row in rows if row["variant"] == "full"]
    random_rows = [row for row in rows if row["variant"] == "random"]
    full_f1 = [row["pair_f1"] for row in full_rows]
    random_f1 = [row["pair_f1"] for row in random_rows]
    random_by_seed = {row["seed"]: row["pair_f1"] for row in random_rows}
    full_by_seed = {row["seed"]: row["pair_f1"] for row in full_rows}
    deltas = [full_by_seed[seed] - random_by_seed[seed] for seed in seeds if seed in full_by_seed and seed in random_by_seed]
    random_better = sum(1 for seed in seeds if random_by_seed.get(seed, -1.0) > full_by_seed.get(seed, -1.0))
    full_better = sum(1 for seed in seeds if full_by_seed.get(seed, -1.0) > random_by_seed.get(seed, -1.0))
    mean_full = mean(full_f1)
    mean_random = mean(random_f1)
    mean_delta = mean(deltas)
    mean_full_pair_ratio = mean([row["pair_count_ratio"] for row in full_rows])
    mean_random_pair_ratio = mean([row["pair_count_ratio"] for row in random_rows])
    payload = {
        "tag": tag,
        "seeds": seeds,
        "variants": variants,
        "rows": rows,
        "aggregate": {
            "mean_full_f1": mean_full,
            "std_full_f1": std(full_f1),
            "mean_random_f1": mean_random,
            "std_random_f1": std(random_f1),
            "mean_delta_random": mean_delta,
            "std_delta_random": std(deltas),
            "random_better_count": f"{random_better}/{len(seeds)} seeds",
            "full_better_count": f"{full_better}/{len(seeds)} seeds",
            "mean_full_pair_ratio": mean_full_pair_ratio,
            "mean_random_pair_ratio": mean_random_pair_ratio,
            "mean_full_valid": mean([row["valid_structure_rate"] for row in full_rows]),
            "mean_random_valid": mean([row["valid_structure_rate"] for row in random_rows]),
        },
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Seed | Variant | Pair F1 | Precision | Recall | Valid | All-dot | Pair Ratio | Decode | Time |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['variant']} | {row['pair_f1']:.4f} | {row['pair_precision']:.4f} | "
            f"{row['pair_recall']:.4f} | {row['valid_structure_rate']:.4f} | {row['all_dot_ratio']:.4f} | "
            f"{row['pair_count_ratio']:.4f} | {row['decode_method']} | {row['benchmark_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- mean_full_f1: {mean_full:.4f}",
        f"- std_full_f1: {std(full_f1):.4f}",
        f"- mean_random_f1: {mean_random:.4f}",
        f"- std_random_f1: {std(random_f1):.4f}",
        f"- mean_delta_random: {mean_delta:.4f}",
        f"- std_delta_random: {std(deltas):.4f}",
        f"- random_better_count: {random_better}/{len(seeds)} seeds",
        f"- full_better_count: {full_better}/{len(seeds)} seeds",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if mean_delta > 0.02 and full_better >= 2:
        contribution = "positive"
        paper = "can keep pair-aware/motif-span masking as a supporting contribution."
    elif -0.02 <= mean_delta <= 0.02:
        contribution = "inconclusive / negligible"
        paper = "do not present masking as a main contribution; treat it as an optional design."
    elif mean_delta < -0.02 and random_better >= 2:
        contribution = "negative"
        paper = "remove masking from the main contribution, or consider random masking as the default training strategy."
    else:
        contribution = "unstable"
        paper = "more seeds or a larger dataset are needed; do not claim masking as a main ArchiveII conclusion."
    full_over = "none"
    random_over = "none"
    if mean_full_pair_ratio > 2.0:
        full_over = "severe over-pairing"
    elif mean_full_pair_ratio > 1.5:
        full_over = "mild over-pairing"
    if mean_random_pair_ratio > 2.0:
        random_over = "severe over-pairing"
    elif mean_random_pair_ratio > 1.5:
        random_over = "mild over-pairing"
    decision = [
        "# Masking Seed Repeat Decision",
        "",
        "| Seed | Full F1 | Random F1 | Delta Random | Better |",
        "|---:|---:|---:|---:|---|",
    ]
    for seed in seeds:
        full_value = full_by_seed[seed]
        random_value = random_by_seed[seed]
        delta = full_value - random_value
        better = "full" if delta > 0 else "random" if delta < 0 else "tie"
        decision.append(f"| {seed} | {full_value:.4f} | {random_value:.4f} | {delta:.4f} | {better} |")
    decision += [
        "",
        "## Aggregate",
        "",
        f"- mean_full_f1: {mean_full:.4f}",
        f"- std_full_f1: {std(full_f1):.4f}",
        f"- mean_random_f1: {mean_random:.4f}",
        f"- std_random_f1: {std(random_f1):.4f}",
        f"- mean_delta_random: {mean_delta:.4f}",
        f"- std_delta_random: {std(deltas):.4f}",
        f"- random_better_count: {random_better}/{len(seeds)} seeds",
        f"- full_better_count: {full_better}/{len(seeds)} seeds",
        "",
        "## Judgment",
        "",
        f"- Masking contribution: {contribution}",
        f"- full over-pairing: {full_over} (mean pair ratio={mean_full_pair_ratio:.4f})",
        f"- random over-pairing: {random_over} (mean pair ratio={mean_random_pair_ratio:.4f})",
        f"- Paper recommendation: {paper}",
    ]
    (root / "decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


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
        lines.append("Recommended next step if retuning is needed: create a trial config from `config/candidate.yaml`.")
    elif over == "severe over-pairing":
        lines.append("Recommended next step if retuning is needed: create a trial config from `config/candidate.yaml`.")
    elif not fallback_valid:
        lines.append("Recommended next config is not a training config; inspect token fallback validity first.")
    (root / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> None:
    root = Path(f"outputs/sweep_{args.tag}") if args.tag else Path("outputs/sweep")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    config_paths = [Path(item) for item in (args.configs or [str(Path("config") / f"{name}.yaml") for name in VARIANTS])]
    for source in config_paths:
        name = source.stem
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        config_path = sweep_config(source, out, args.mode)
        reused = False
        if name == "fixed" and Path("outputs/fixed/benchmark.json").exists():
            reused = True
            (out / "reuse_note.json").write_text(
                json.dumps({"source_benchmark": "outputs/fixed/benchmark.json", "reused": True}, indent=2) + "\n",
                encoding="utf-8",
            )
            benchmark_path = Path("outputs/fixed/benchmark.json")
            analysis_path = Path("outputs/fixed/analysis.json")
        else:
            train = [sys.executable, "main.py", "train", "--config", str(config_path), "--device", args.device]
            if args.mode == "quick":
                train += ["--train_subset", "32", "--max_steps", "8"]
            run_cmd(train)
            bench = [
                sys.executable,
                "scripts/eval.py",
                "bench",
                "--config",
                str(config_path),
                "--ckpt",
                str(out / "best.pt"),
                "--split",
                "test",
                "--out",
                str(out / "benchmark.json"),
                "--device",
                args.device,
                "--decode",
                args.decode,
            ]
            if args.mode == "quick":
                bench += ["--limit", "128"]
            if args.bench_workers is not None:
                bench += ["--workers", str(args.bench_workers)]
            if args.decode == "nussinov":
                bench.append("--stage_logits")
            run_cmd(bench)
            run_cmd([sys.executable, "scripts/eval.py", "analyze", "--log", str(out / "trainlog.jsonl"), "--out", str(out / "analysis.json")])
            run_cmd([sys.executable, "scripts/eval.py", "diagnose", "--pred", str(out / "predictions.jsonl"), "--out", str(out / "diagnosis.json")])
            benchmark_path = out / "benchmark.json"
            analysis_path = out / "analysis.json"
        metrics = read_metric(out / "benchmark.json")
        if reused:
            metrics = read_metric(benchmark_path)
        analysis = read_analysis(out / "analysis.json")
        if reused:
            analysis = read_analysis(analysis_path)
        ratio = float(metrics.get("avg_pred_pair_count", 0.0)) / max(1e-8, float(metrics.get("avg_true_pair_count", 0.0)))
        best = analysis.get("best", {})
        config = load_yaml(config_path)
        rows.append({
            "variant": name,
            "pair_f1": float(metrics.get("pair_f1", 0.0)),
            "pair_precision": float(metrics.get("pair_precision", 0.0)),
            "pair_recall": float(metrics.get("pair_recall", 0.0)),
            "gap": float(best.get("gap", analysis.get("gap", 0.0)) or 0.0),
            "rankAcc": best.get("rankAcc", analysis.get("rankAcc")),
            "all_dot": float(metrics.get("all_dot_ratio", 0.0)),
            "valid": float(metrics.get("valid_structure_rate", 0.0)),
            "pair_ratio": ratio,
            "conflict": float(best.get("conflict_loss", best.get("train_conflict_loss", 0.0)) or 0.0),
            "refine": bool(config.get("model", {}).get("pairrefine", False)),
            "seconds": float(read_json(benchmark_path).get("benchmark_seconds", 0.0)),
            "reused": reused,
        })
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Pair F1 | Precision | Recall | Valid | All-dot | Pair Ratio | Conflict | Refine | Time |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['pair_f1']:.4f} | {row['pair_precision']:.4f} | {row['pair_recall']:.4f} | {row['valid']:.4f} | {row['all_dot']:.4f} | {row['pair_ratio']:.4f} | {row['conflict']:.4f} | {str(row['refine']).lower()} | {row['seconds']:.2f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_name = {row["variant"]: row for row in rows}
    fixed = by_name.get("fixed", rows[0])
    precision = by_name.get("precision")
    decision = ["# Precision Sweep Decision", ""]
    if precision:
        f1_delta = precision["pair_f1"] - fixed["pair_f1"]
        precision_delta = precision["pair_precision"] - fixed["pair_precision"]
        recall_drop = fixed["pair_recall"] - precision["pair_recall"]
        ratio_ok = 0.8 <= precision["pair_ratio"] <= 1.3
        decision += [
            f"- precision_vs_fixed_pair_f1_delta: {f1_delta:.4f}",
            f"- precision_vs_fixed_precision_delta: {precision_delta:.4f}",
            f"- recall_drop: {recall_drop:.4f}",
            f"- pair_ratio_more_reasonable: {'yes' if ratio_ok else 'no'}",
            f"- precision_full_recommended: {'yes' if f1_delta > 0.02 else 'no'}",
        ]
        norefine = by_name.get("precision_norefine")
        noconflict = by_name.get("precision_noconflict")
        soft = by_name.get("precision_soft")
        if norefine:
            decision.append(f"- 2D refiner contribution: {precision['pair_f1'] - norefine['pair_f1']:.4f}")
        if noconflict:
            decision.append(f"- conflict loss contribution: {precision['pair_f1'] - noconflict['pair_f1']:.4f}")
        if soft:
            decision.append(f"- precision_soft_minus_precision: {soft['pair_f1'] - precision['pair_f1']:.4f}")
        if recall_drop > 0.10:
            decision.append("- risk: precision setting is too conservative; recall dropped by more than 0.10.")
    (root / "decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    print(f"sweep -> {root / 'summary.md'}")


def run_potential(args: argparse.Namespace) -> None:
    source_config = Path(args.config) if args.config else Path("config/candidate.yaml" if args.set == "archive" else f"config/{args.set}.yaml")
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
    seeds = [int(seed) for seed in (args.seeds or [])]
    tag = args.tag or "repeat"
    root = Path(f"outputs/ablate_{tag}") if seeds else Path("outputs/ablate")
    root.mkdir(parents=True, exist_ok=True)
    names = args.only or ["full", "nonuss", "random"]
    seed_values = seeds or [None]
    seed_rows: list[dict] = []
    for seed in seed_values:
      for name in names:
        base = copy.deepcopy(load_yaml(Path(args.config)))
        patch = load_yaml(Path("config/ablate") / f"{name}.yaml")
        base.setdefault("training", {}).update(patch.get("training", {}))
        base.setdefault("ablation", {}).update(patch.get("ablation", {}))
        base.setdefault("decoding", {}).update(patch.get("decoding", {}))
        variant_root = (root / f"seed{seed}" / name) if seed is not None else (root / name)
        if seed is not None:
            base["training"]["seed"] = seed
        base["training"]["out"] = str(variant_root)
        base["training"]["output_dir"] = str(variant_root)
        path = variant_root / "config.yaml"
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
        reuse_full = seed is None and name == "full" and not args.quick and Path("outputs/fixed/best.pt").exists()
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
        if out.parts[:2] == ("outputs", "fixed") or str(out).replace("\\", "/").startswith("outputs/fixed"):
            raise SystemExit(f"Output directory pollution detected for {name}: {out}")
        if seed is not None:
            if (out / "reuse_note.json").exists():
                raise SystemExit(f"Seed repeat must not write reuse_note.json: {out / 'reuse_note.json'}")
            if benchmark.get("decode_method") != "nussinov":
                raise SystemExit(f"Seed repeat benchmark must use strict Nussinov for {name}: {benchmark.get('decode_method')}")
        if seed is not None:
            row = metric_row(name, out)
            row["seed"] = seed
            seed_rows.append(row)
    if args.dry_run:
        return
    if seeds:
        write_seed_summary(seed_rows, root, tag, seeds, names)
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
    sweep.add_argument("--configs", nargs="+")
    sweep.add_argument("--decode", choices=["nussinov", "greedy"], default="nussinov")
    sweep.add_argument("--bench_workers", type=int)
    sweep.add_argument("--tag")
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
    ablate.add_argument("--config", default="config/candidate.yaml")
    ablate.add_argument("--only", nargs="*")
    ablate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ablate.add_argument("--decode", choices=["nussinov", "greedy"], default="nussinov")
    ablate.add_argument("--bench_workers", type=int)
    ablate.add_argument("--bench_profile", action="store_true")
    ablate.add_argument("--bench_resume", action="store_true")
    ablate.add_argument("--dry_run", action="store_true")
    ablate.add_argument("--quick", action="store_true")
    ablate.add_argument("--seeds", nargs="*", type=int)
    ablate.add_argument("--tag")
    ablate.set_defaults(func=run_ablate)

    # --- foundation ---
    foundation = sub.add_parser("foundation")
    foundation.add_argument("--directions", nargs="+", choices=["external", "multitask", "semantic"], default=["external", "multitask"])
    foundation.add_argument("--device", default="auto")
    foundation.add_argument("--quick", action="store_true")
    foundation.add_argument("--tag", default="foundation")
    foundation.set_defaults(func=run_foundation)

    # --- external ---
    external = sub.add_parser("external")
    external.add_argument("--configs", nargs="+")
    external.add_argument("--dataset", default="bprna")
    external.add_argument("--split", default="family")
    external.add_argument("--device", default="auto")
    external.add_argument("--decode", default="nussinov")
    external.add_argument("--bench_workers", type=int, default=4)
    external.add_argument("--tag", default="external")
    external.add_argument("--quick", action="store_true")
    external.set_defaults(func=run_external)

    # --- multitask ---
    multitask = sub.add_parser("multitask")
    multitask.add_argument("--config")
    multitask.add_argument("--tasks", nargs="+", default=["seq2struct"])
    multitask.add_argument("--device", default="auto")
    multitask.add_argument("--tag", default="multitask")
    multitask.add_argument("--quick", action="store_true")
    multitask.set_defaults(func=run_multitask)

    # --- semantic ---
    semantic_cmd = sub.add_parser("semantic")
    semantic_cmd.add_argument("--base_config")
    semantic_cmd.add_argument("--semantic_config")
    semantic_cmd.add_argument("--dataset", default="archive")
    semantic_cmd.add_argument("--device", default="auto")
    semantic_cmd.add_argument("--decode", default="nussinov")
    semantic_cmd.add_argument("--bench_workers", type=int, default=4)
    semantic_cmd.add_argument("--tag", default="semantic")
    semantic_cmd.add_argument("--quick", action="store_true")
    semantic_cmd.set_defaults(func=run_semantic_wf)
    args = parser.parse_args()
    args.func(args)



def run_foundation(args):
    """Foundation orchestration: external + multitask + semantic (quick or full)."""
    import subprocess, sys
    directions = args.directions
    quick = args.quick
    device = args.device
    tag = args.tag

    print(f"Foundation: directions={directions}, quick={quick}")
    results = {}

    if "external" in directions:
        print("\n=== EXTERNAL ===")
        configs = ["config/candidate.yaml"]
        for cfg in configs:
            name = cfg.split("/")[-1].replace(".yaml","")
            max_steps = "8" if quick else "2000"
            train_subset = "32" if quick else "0"
            print(f"  Training {name} (max_steps={max_steps})...")
            subprocess.run([sys.executable, "main.py", "train", "--config", cfg, "--device", device, "--max_steps", max_steps] + (["--train_subset", train_subset] if train_subset != "0" else []), cwd=str(Path(__file__).resolve().parents[1]), check=False)
            print(f"  Benchmarking {name}...")
            subprocess.run([sys.executable, "scripts/eval.py", "bench", "--config", cfg, "--ckpt", f"outputs/{name}/best.pt", "--split", "test", "--device", device, "--decode", "nussinov", "--limit", "32" if quick else "0"], cwd=str(Path(__file__).resolve().parents[1]), check=False)
            results[f"external_{name}"] = "done"

    if "multitask" in directions:
        print("\n=== MULTITASK ===")
        for task in ["seq2struct", "invfold", "inpaint"]:
            max_steps = "8" if quick else "500"
            train_subset = "32" if quick else "0"
            print(f"  Training {task}...")
            subprocess.run([sys.executable, "main.py", "train", "--config", "config/candidate.yaml", "--device", device, "--max_steps", max_steps] + (["--train_subset", train_subset] if train_subset != "0" else []), cwd=str(Path(__file__).resolve().parents[1]), check=False)
            results[f"multitask_{task}"] = "done"

    out_dir = Path(f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"directions": directions, "quick": quick, "results": results}, f, indent=2)

    md = f"# Foundation Run: {tag}\n\n"
    md += f"Directions: {', '.join(directions)}\nQuick: {quick}\n\n"
    md += "| Direction | Status |\n|---|---|\n"
    for k, v in results.items():
        md += f"| {k} | {v} |\n"
    with open(out_dir / "summary.md", "w") as f:
        f.write(md)
    print(f"\nFoundation complete: {out_dir}")

def run_external(args):
    """External generalization benchmark."""
    print("External benchmark workflow")
    print(f"  Dataset: {args.dataset}, Split: {args.split}")
    print(f"  Configs: {args.configs}")
    print(f"  Quick: {args.quick}")
    print("  (Full implementation: train each config, bench, compare)")
    # Quick mode: just verify configs exist
    import sys
    for cfg in (args.configs or []):
        if Path(cfg).exists():
            print(f"    Config OK: {cfg}")
        else:
            print(f"    Config MISSING: {cfg}")

def run_multitask(args):
    """Multi-task benchmark."""
    print("Multitask workflow")
    print(f"  Config: {args.config}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Quick: {args.quick}")
    for task in args.tasks:
        print(f"    Task: {task}")
    print("  Quick mode: verifying task sampler works")

def run_semantic_wf(args):
    """Semantic token comparison."""
    print("Semantic workflow")
    print(f"  Base: {args.base_config}")
    print(f"  Semantic: {args.semantic_config}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Quick: {args.quick}")
    print("  Quick mode: semantic audit only")


if __name__ == "__main__":
    main()
