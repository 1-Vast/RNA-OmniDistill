from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSON at {path}:{line_no}: {exc}")
    return rows


def trend(values: list[float]) -> dict[str, float | bool]:
    if not values:
        return {"first": 0.0, "last": 0.0, "best": 0.0, "delta": 0.0, "decreased": False, "increased": False}
    first = float(values[0])
    last = float(values[-1])
    return {
        "first": first,
        "last": last,
        "best": max(values),
        "min": min(values),
        "delta": last - first,
        "decreased": last < first,
        "increased": last > first,
    }


def series(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def diagnose(rows: list[dict]) -> tuple[list[str], list[str]]:
    issues = []
    suggestions = []
    last = rows[-1]
    train_loss = series(rows, "train_loss")
    val_pair_f1 = series(rows, "val_pair_f1")
    val_all_dot = float(last.get("val_all_dot_ratio", 0.0))
    avg_pred = float(last.get("val_avg_pred_pair_count", 0.0))
    avg_true = float(last.get("val_avg_true_pair_count", 0.0))
    pos_logit = float(last.get("positive_pair_logit_mean", 0.0))
    neg_logit = float(last.get("negative_pair_logit_mean", 0.0))
    token_acc = series(rows, "val_token_acc")

    if val_all_dot > 0.8:
        issues.append("all-dot collapse: val_all_dot_ratio > 0.8")
        suggestions.extend(
            [
                "increase lambda_pair",
                "increase structure_positive_weight",
                "check pair labels",
                "check whether Nussinov scoring is too conservative",
                "lower min_loop_length only for debug",
                "check true all-dot ratio in the training set",
            ]
        )
    if avg_true > 0 and avg_pred < 0.2 * avg_true:
        issues.append("under-pairing: avg_pred_pair_count is far below avg_true_pair_count")
        suggestions.extend(["increase lambda_pair", "increase pair_positive_weight", "inspect pair head logits"])
    if avg_true > 0 and avg_pred > 2.0 * avg_true:
        issues.append("over-pairing: avg_pred_pair_count is more than 2x avg_true_pair_count")
        suggestions.extend(
            [
                "decrease lambda_pair",
                "increase negative sampling ratio",
                "increase Nussinov penalty or threshold",
                "check whether pair_positive_weight is too large",
            ]
        )
    if pos_logit <= neg_logit:
        issues.append("pair head has not learned positive-vs-negative ordering")
        suggestions.extend(
            [
                "check pair_labels and pair_mask alignment",
                "check seq_positions map only to sequence tokens",
                "verify pair logits are not polluted by padding or special tokens",
            ]
        )
    if train_loss and val_pair_f1 and train_loss[-1] < train_loss[0] and max(val_pair_f1) <= val_pair_f1[0] + 1e-6:
        issues.append("train_loss decreased but val_pair_f1 did not improve")
        suggestions.extend(
            [
                "possible token overfitting or pair decoding mismatch",
                "do not rely only on dot-bracket token accuracy",
                "check Nussinov input scores come from correct pair probabilities",
            ]
        )
    if token_acc and val_pair_f1 and token_acc[-1] > 0.7 and val_pair_f1[-1] < 0.2:
        issues.append("token accuracy is high but pair F1 is low")
        suggestions.extend(
            [
                "do not over-rely on dot-bracket token head",
                "check pair head hidden states correspond to sequence positions",
                "check Nussinov score source",
            ]
        )
    if val_pair_f1 and max(val_pair_f1) == 0.0:
        issues.append("val_pair_f1 stayed at 0")
        suggestions.extend(["inspect pair labels", "increase pair loss pressure", "check all-dot true structure ratio"])
    if not issues:
        issues.append("no major automatic failure detected")
    return issues, sorted(set(suggestions))


def write_text(path: Path, analysis: dict) -> None:
    lines = [
        "RNA-OmniDiffusion Training Analysis",
        "",
        f"epochs_logged: {analysis['epochs_logged']}",
        f"best_epoch: {analysis['best_epoch']}",
        f"best_val_pair_f1: {analysis['best_val_pair_f1']:.4f}",
        f"best_checkpoint_path: {analysis['best_checkpoint_path']}",
        f"early_stopping_inferred: {analysis['early_stopping_inferred']}",
        "",
        "Curves:",
    ]
    for key, value in analysis["curves"].items():
        lines.append(f"- {key}: first={value.get('first', 0):.4f} last={value.get('last', 0):.4f} delta={value.get('delta', 0):.4f}")
    lines.append("")
    lines.append("Diagnosis:")
    lines.extend(f"- {item}" for item in analysis["diagnosis"])
    lines.append("")
    lines.append("Suggestions:")
    lines.extend(f"- {item}" for item in analysis["suggestions"] or ["none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RNA-OmniDiffusion train_log.jsonl.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Training log not found: {log_path}. Run training first or pass a valid --log path.")
    rows = read_log(log_path)
    if not rows:
        raise SystemExit(f"Training log is empty: {log_path}")

    best_row = max(rows, key=lambda row: float(row.get("val_pair_f1", 0.0)))
    issues, suggestions = diagnose(rows)
    curves = {
        "train_loss": trend(series(rows, "train_loss")),
        "val_loss": trend(series(rows, "val_loss")),
        "val_pair_f1": trend(series(rows, "val_pair_f1")),
        "val_token_acc": trend(series(rows, "val_token_acc")),
        "val_all_dot_ratio": trend(series(rows, "val_all_dot_ratio")),
        "val_avg_pred_pair_count": trend(series(rows, "val_avg_pred_pair_count")),
        "val_avg_true_pair_count": trend(series(rows, "val_avg_true_pair_count")),
        "positive_pair_logit_mean": trend(series(rows, "positive_pair_logit_mean")),
        "negative_pair_logit_mean": trend(series(rows, "negative_pair_logit_mean")),
    }
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(rows)]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = {
        "epochs_logged": len(rows),
        "first_epoch": min(epochs),
        "last_epoch": max(epochs),
        "early_stopping_inferred": len(rows) >= 2 and max(epochs) < int(rows[-1].get("configured_epochs", max(epochs))),
        "best_epoch": int(best_row.get("epoch", 0)),
        "best_val_pair_f1": float(best_row.get("val_pair_f1", 0.0)),
        "best_checkpoint_path": str(log_path.parent / "best.pt"),
        "curves": curves,
        "diagnosis": issues,
        "suggestions": suggestions,
    }
    output_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    txt_path = output_path.with_suffix(".txt")
    write_text(txt_path, analysis)
    print(f"analysis json -> {output_path}")
    print(f"analysis txt -> {txt_path}")
    for issue in issues:
        print(f"diagnosis: {issue}")


if __name__ == "__main__":
    main()

