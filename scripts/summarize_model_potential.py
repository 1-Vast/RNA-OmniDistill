from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SystemExit(f"Required file not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(data: dict, method: str, key: str) -> float:
    return float(data.get("overall", {}).get(method, {}).get(key, 0.0))


def ablation_delta(summary: dict, variant: str, full_f1: float) -> float:
    rows = summary.get("rows", [])
    for row in rows:
        if row.get("variant") == variant:
            return full_f1 - float(row.get("pair_f1", 0.0))
    return 0.0


def choose_potential(metrics: dict, failures: list[str]) -> tuple[str, list[str]]:
    reasons = []
    beats_baseline = metrics["test_pair_f1"] > metrics["random_valid_pair_baseline_f1"] + 1e-6
    pair_ordered = metrics["pair_logit_gap"] > 0
    pair_count_ok = 0.5 <= metrics["pair_count_ratio"] <= 1.5
    ablation_available = any(
        metrics[key] != 0.0
        for key in ["full_vs_no_pair_head_delta", "full_vs_no_nussinov_delta", "full_vs_random_mask_only_delta"]
    )
    ablation_helpful = ablation_available and (
        metrics["full_vs_no_pair_head_delta"] > 0
        or metrics["full_vs_no_nussinov_delta"] > 0
        or metrics["full_vs_random_mask_only_delta"] > 0
    )
    if beats_baseline:
        reasons.append("model beats random valid-pair baseline")
    if pair_ordered:
        reasons.append("positive pair logits exceed negative pair logits")
    if metrics["all_dot_ratio"] < 0.3:
        reasons.append("no all-dot collapse")
    if metrics["valid_structure_rate"] >= 0.95:
        reasons.append("decoded structures are mostly valid")
    if pair_count_ok:
        reasons.append("predicted pair count is in a reasonable range")
    if ablation_helpful:
        reasons.append("full model improves over at least one core ablation")

    if beats_baseline and pair_ordered and metrics["all_dot_ratio"] < 0.3 and metrics["valid_structure_rate"] >= 0.95 and pair_count_ok:
        return "High", reasons
    if beats_baseline and metrics["all_dot_ratio"] < 0.8 and failures:
        return "Medium", reasons + ["failures are present but diagnosable"]
    return "Low", reasons or ["model does not beat baseline or pair head signal is weak"]


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Model Potential Summary",
        "",
        f"Conclusion: **{result['potential']}**",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in result["metrics"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Evidence"])
    lines.extend(f"- {item}" for item in result["reasons"])
    lines.extend(["", "## Failure Risks"])
    lines.extend(f"- {item}" for item in result["failure_risks"] or ["none detected"])
    lines.extend(
        [
            "",
            "## Recommendations",
            f"- Expand to RNAStrAlign/bpRNA: {'yes' if result['potential'] in {'High', 'Medium'} else 'not yet'}",
            f"- Run more ablations: {'yes' if result['potential'] in {'High', 'Medium'} else 'only after fixing core failures'}",
            f"- Agent work: {'defer framework choice; use diagnostics first' if result['potential'] != 'High' else 'still defer framework choice until full data confirms'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whether the base model has research potential.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--training_analysis", required=True)
    parser.add_argument("--prediction_diagnosis", required=True)
    parser.add_argument("--ablation_summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    benchmark = read_json(Path(args.benchmark))
    training = read_json(Path(args.training_analysis), required=False)
    diagnosis = read_json(Path(args.prediction_diagnosis), required=False)
    ablations = read_json(Path(args.ablation_summary), required=False)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_f1 = metric(benchmark, "model", "pair_f1")
    avg_true = metric(benchmark, "model", "avg_true_pair_count")
    metrics = {
        "test_pair_precision": metric(benchmark, "model", "pair_precision"),
        "test_pair_recall": metric(benchmark, "model", "pair_recall"),
        "test_pair_f1": full_f1,
        "all_dot_ratio": metric(benchmark, "model", "all_dot_ratio"),
        "valid_structure_rate": metric(benchmark, "model", "valid_structure_rate"),
        "canonical_pair_ratio": metric(benchmark, "model", "canonical_pair_ratio"),
        "avg_pred_pair_count": metric(benchmark, "model", "avg_pred_pair_count"),
        "avg_true_pair_count": avg_true,
        "pair_count_ratio": metric(benchmark, "model", "avg_pred_pair_count") / max(1e-8, avg_true),
        "positive_pair_logit_mean": float(training.get("curves", {}).get("positive_pair_logit_mean", {}).get("last", 0.0)),
        "negative_pair_logit_mean": float(training.get("curves", {}).get("negative_pair_logit_mean", {}).get("last", 0.0)),
        "random_valid_pair_baseline_f1": metric(benchmark, "random_pair", "pair_f1"),
        "all_dot_baseline_f1": metric(benchmark, "all_dot", "pair_f1"),
        "full_vs_no_pair_head_delta": ablation_delta(ablations, "no_pair_head", full_f1),
        "full_vs_no_nussinov_delta": ablation_delta(ablations, "no_nussinov", full_f1),
        "full_vs_random_mask_only_delta": ablation_delta(ablations, "random_mask_only", full_f1),
    }
    metrics["pair_logit_gap"] = metrics["positive_pair_logit_mean"] - metrics["negative_pair_logit_mean"]
    failure_risks = diagnosis.get("suggestions", []) or diagnosis.get("diagnosis", []) or []
    potential, reasons = choose_potential(metrics, failure_risks)
    result = {
        "potential": potential,
        "metrics": metrics,
        "reasons": reasons,
        "failure_risks": failure_risks,
        "toy_fallback_only": bool(benchmark.get("toy_fallback_only", False)),
    }
    if result["toy_fallback_only"]:
        result["potential"] = "Low"
        result["failure_risks"].append("toy fallback only; not valid for model potential conclusion")

    (output_dir / "model_potential.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "model_potential.md", result)
    with (output_dir / "model_potential.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["potential", result["potential"]])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"model potential -> {output_dir / 'model_potential.md'}")
    print(f"conclusion={result['potential']}")


if __name__ == "__main__":
    main()
