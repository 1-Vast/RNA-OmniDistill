from __future__ import annotations

import json
import math
import os
import statistics
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from utils.struct import canonical_pair, parse_dot_bracket, validate_structure


ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You are an RNA secondary-structure experiment analysis agent.
You analyze logs, metrics, dataset summaries, and paper text.
You must not modify labels, benchmark metrics, predictions, or test data.
You must not participate in benchmark inference.
You must not claim LLMs improve RNA-OmniDiffusion model performance.
You must not invent files, commands, datasets, metrics, or baselines.
Use only repository commands with main.py, scripts/eval.py, scripts/run.py, scripts/data.py, scripts/audit.py, scripts/probe.py, and scripts/llm.py.
Do not cite external performance thresholds unless they are supplied in the user prompt.
Treat every output as experiment assistance, not as model output.
Be concise, technical, and explicit about uncertainty."""


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value
    return env


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def read_text(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_report(out: Path, stem: str, data: dict[str, Any], markdown: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if data.get("dry_run"):
        (out / "request.json").write_text(json.dumps({**data, "prompt": markdown}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "prompt.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {out / 'request.json'}")
        print(f"Wrote {out / 'prompt.md'}")
        return
    (out / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{stem}.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out / (stem + '.json')}")
    print(f"Wrote {out / (stem + '.md')}")


def compact_run(run_dir: Path) -> dict[str, Any]:
    benchmark = read_json(run_dir / "benchmark.json")
    analysis = read_json(run_dir / "analysis.json")
    diagnosis = read_json(run_dir / "diagnosis.json")
    train_rows = read_jsonl(run_dir / "trainlog.jsonl")
    last_train = train_rows[-1] if train_rows else {}
    model = benchmark.get("overall", {}).get("model", {})
    random_base = benchmark.get("overall", {}).get("random", {})
    avg_pred = float(model.get("avg_pred_pair_count") or 0.0)
    avg_true = float(model.get("avg_true_pair_count") or 0.0)
    return {
        "run_dir": str(run_dir),
        "pair_f1": model.get("pair_f1"),
        "pair_precision": model.get("pair_precision"),
        "pair_recall": model.get("pair_recall"),
        "valid_structure_rate": model.get("valid_structure_rate"),
        "all_dot_ratio": model.get("all_dot_ratio"),
        "avg_pred_pair_count": avg_pred,
        "avg_true_pair_count": avg_true,
        "pair_count_ratio": avg_pred / avg_true if avg_true else None,
        "random_pair_f1": random_base.get("pair_f1"),
        "decode_method": benchmark.get("decode_method"),
        "last_gap": last_train.get("gap"),
        "last_rankAcc": last_train.get("rankAcc"),
        "last_loss": last_train.get("loss"),
        "analysis_keys": sorted(analysis.keys()),
        "diagnosis_keys": sorted(diagnosis.keys()),
        "train_log_rows": len(train_rows),
    }


def finite_number(value: Any) -> bool:
    try:
        item = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(item)


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        item = float(value)
    except (TypeError, ValueError):
        return default
    return item if math.isfinite(item) else default


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def metric_block(benchmark: dict[str, Any]) -> dict[str, Any]:
    return benchmark.get("overall", {}).get("model", {}) if benchmark else {}


def pair_ratio(metrics: dict[str, Any]) -> float | None:
    pred = as_float(metrics.get("avg_pred_pair_count"))
    true = as_float(metrics.get("avg_true_pair_count"))
    if pred is None or true is None or true == 0:
        return None
    return pred / true


def summarize_curve(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    values = [as_float(first_present(row, names)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {"available": False}
    return {
        "available": True,
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "delta": values[-1] - values[0],
        "decreased": values[-1] < values[0],
        "count": len(values),
    }


def run_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "trainlog": run_dir / "trainlog.jsonl",
        "train_log": run_dir / "train_log.jsonl",
        "benchmark": run_dir / "benchmark.json",
        "benchmark_csv": run_dir / "benchmark.csv",
        "predictions": run_dir / "predictions.jsonl",
        "runmeta": run_dir / "runmeta.json",
        "config": run_dir / "config.yaml",
        "best_ckpt": run_dir / "best.pt",
        "last_ckpt": run_dir / "last.pt",
    }


def config_flags(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    decoding = config.get("decoding", {})
    ablation = config.get("ablation", {})
    return {
        "pairrefine": bool(model.get("pairrefine", False)),
        "pairhead": model.get("pairhead"),
        "use_nussinov": bool(decoding.get("use_nussinov", ablation.get("use_nussinov", False))),
        "decode_source": decoding.get("decode_source", decoding.get("source")),
        "pair_threshold": decoding.get("pair_threshold", decoding.get("threshold")),
        "nussinov_gamma": decoding.get("nussinov_gamma", decoding.get("gamma")),
        "pair_prior": config.get("pair_prior", config.get("pairprior", None)),
    }


def inspect_run_artifacts(run_dir: Path) -> dict[str, Any]:
    paths = run_artifacts(run_dir)
    train_path = paths["trainlog"] if paths["trainlog"].exists() else paths["train_log"]
    train_rows = read_jsonl(train_path)
    benchmark = read_json(paths["benchmark"])
    metrics = metric_block(benchmark)
    config = read_json(paths["runmeta"]).get("config", {}) if paths["runmeta"].exists() else {}
    if not config and paths["config"].exists():
        try:
            import yaml

            config = yaml.safe_load(paths["config"].read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}
    flags = config_flags(config)
    loss_curve = summarize_curve(train_rows, ["loss", "train_loss"])
    pair_curve = summarize_curve(train_rows, ["pair_loss", "train_pair_loss"])
    token_curve = summarize_curve(train_rows, ["token_loss", "train_token_loss"])
    gap = as_float(first_present(train_rows[-1], ["gap", "pair_logit_gap"])) if train_rows else None
    rank = as_float(first_present(train_rows[-1], ["rankAcc", "pair_ranking_accuracy_sampled"])) if train_rows else None
    ratio = pair_ratio(metrics)
    nonfinite = []
    for idx, row in enumerate(train_rows):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not finite_number(value):
                nonfinite.append({"row": idx, "key": key, "value": value})
    warnings = []
    if loss_curve.get("available") and not loss_curve.get("decreased"):
        warnings.append("train loss did not decrease")
    if pair_curve.get("available") and not pair_curve.get("decreased"):
        warnings.append("pair loss did not decrease")
    if nonfinite:
        warnings.append("NaN or Inf found in train log")
    if gap is not None and gap <= 0:
        warnings.append("pair logit gap is not positive")
    if rank is not None and rank <= 0.5:
        warnings.append("rankAcc is not above random")
    if ratio is not None and not (0.5 <= ratio <= 1.5):
        warnings.append("pair count ratio outside [0.5, 1.5]")
    valid = as_float(metrics.get("valid_structure_rate"))
    if valid is not None and valid < 1.0:
        warnings.append("valid structure rate below 1.0")
    all_dot = as_float(metrics.get("all_dot_ratio"))
    if all_dot is not None and all_dot > 0.3:
        warnings.append("all-dot ratio is high")
    if not paths["best_ckpt"].exists():
        warnings.append("best checkpoint missing")
    decode_method = benchmark.get("decode_method")
    if flags.get("use_nussinov") and decode_method and decode_method != "nussinov":
        warnings.append("config expects Nussinov but benchmark decode_method differs")
    return {
        "run_dir": str(run_dir),
        "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in paths.items()},
        "train_rows": len(train_rows),
        "loss_curve": loss_curve,
        "pair_loss_curve": pair_curve,
        "token_loss_curve": token_curve,
        "nonfinite_count": len(nonfinite),
        "nonfinite_examples": nonfinite[:20],
        "pair_logit_gap": gap,
        "rankAcc": rank,
        "pair_count_ratio": ratio,
        "metrics": metrics,
        "checkpoint": {
            "best_exists": paths["best_ckpt"].exists(),
            "last_exists": paths["last_ckpt"].exists(),
            "save_best_by": config.get("training", {}).get("save_best_by"),
        },
        "config_flags": flags,
        "decode_method": decode_method,
        "warnings": warnings,
        "status": "PASS" if not warnings else "WARN",
    }


def inspect_markdown(result: dict[str, Any]) -> list[str]:
    metrics = result.get("metrics", {})
    return [
        "# Run Inspection",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Training",
        f"- train rows: {result['train_rows']}",
        f"- loss decreased: {result['loss_curve'].get('decreased')}",
        f"- pair loss decreased: {result['pair_loss_curve'].get('decreased')}",
        f"- token loss available: {result['token_loss_curve'].get('available')}",
        f"- nonfinite count: {result['nonfinite_count']}",
        "",
        "## Pair Head",
        f"- pair logit gap: {result.get('pair_logit_gap')}",
        f"- rankAcc: {result.get('rankAcc')}",
        f"- pair count ratio: {result.get('pair_count_ratio')}",
        "",
        "## Benchmark",
        f"- decode method: {result.get('decode_method')}",
        f"- Pair F1: {metrics.get('pair_f1')}",
        f"- Valid rate: {metrics.get('valid_structure_rate')}",
        f"- All-dot ratio: {metrics.get('all_dot_ratio')}",
        "",
        "## Config",
        f"- pairrefine: {result['config_flags'].get('pairrefine')}",
        f"- strict Nussinov: {result['config_flags'].get('use_nussinov')}",
        f"- save_best_by: {result['checkpoint'].get('save_best_by')}",
        "",
        "## Warnings",
        *([f"- {item}" for item in result["warnings"]] if result["warnings"] else ["- none"]),
    ]


def trace_provenance(config_path: Path, ckpt_path: Path, benchmark_path: Path) -> dict[str, Any]:
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    except Exception:
        config = {}
    benchmark = read_json(benchmark_path)
    metrics = metric_block(benchmark)
    flags = config_flags(config or {})
    benchmeta_path = benchmark_path.parent / "benchmeta.json"
    benchmeta = read_json(benchmeta_path)
    pred_path = benchmark_path.parent / "predictions.jsonl"
    pred_rows = read_jsonl(pred_path, limit=100000)
    mismatch_count = 0
    for row in pred_rows:
        seq = row.get("seq", "")
        pred = row.get("pred_struct", "")
        true = row.get("true_struct", row.get("struct", ""))
        if pred and len(seq) != len(pred):
            mismatch_count += 1
        if true and len(seq) != len(true):
            mismatch_count += 1
    warnings = []
    if not ckpt_path.exists():
        warnings.append("checkpoint missing")
    if not benchmark_path.exists():
        warnings.append("benchmark missing")
    if benchmark.get("decode_method") != "nussinov":
        warnings.append("benchmark is not strict Nussinov")
    if flags.get("use_nussinov") and benchmark.get("decode_method") not in {None, "nussinov"}:
        warnings.append("config and benchmark decode path mismatch")
    if mismatch_count:
        warnings.append("prediction length mismatch detected")
    if flags.get("pair_prior"):
        warnings.append("pair-prior appears configured")
    release = read_text(ROOT / "release" / "results_summary.md", max_chars=50000)
    f1 = metrics.get("pair_f1")
    mismatch_risk = False
    if f1 is not None and release:
        f1_text = f"{float(f1):.4f}" if as_float(f1) is not None else str(f1)
        mismatch_risk = f1_text not in release
        if mismatch_risk:
            warnings.append("PROVENANCE_MISMATCH_RISK")
    return {
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "benchmark": str(benchmark_path),
        "config_exists": config_path.exists(),
        "checkpoint_exists": ckpt_path.exists(),
        "benchmark_exists": benchmark_path.exists(),
        "split": benchmeta.get("split", benchmark.get("split")),
        "test_sample_count": metrics.get("samples", benchmark.get("samples")),
        "decode_method": benchmark.get("decode_method"),
        "config_flags": flags,
        "valid_structure_rate": metrics.get("valid_structure_rate"),
        "pair_f1": metrics.get("pair_f1"),
        "prediction_rows": len(pred_rows),
        "prediction_length_mismatches": mismatch_count,
        "official_path_consistent": not warnings,
        "provenance_mismatch_risk": mismatch_risk,
        "warnings": warnings,
        "status": "PASS" if not warnings else "WARN",
    }


def trace_markdown(result: dict[str, Any]) -> list[str]:
    return [
        "# Inference Provenance Trace",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- config: {result['config']} ({result['config_exists']})",
        f"- checkpoint: {result['checkpoint']} ({result['checkpoint_exists']})",
        f"- benchmark: {result['benchmark']} ({result['benchmark_exists']})",
        f"- split: {result.get('split')}",
        f"- test sample count: {result.get('test_sample_count')}",
        f"- decode method: {result.get('decode_method')}",
        f"- pairrefine: {result['config_flags'].get('pairrefine')}",
        f"- use_nussinov: {result['config_flags'].get('use_nussinov')}",
        f"- pair threshold: {result['config_flags'].get('pair_threshold')}",
        f"- nussinov gamma: {result['config_flags'].get('nussinov_gamma')}",
        f"- pair-prior: {result['config_flags'].get('pair_prior')}",
        f"- valid rate: {result.get('valid_structure_rate')}",
        f"- prediction length mismatches: {result.get('prediction_length_mismatches')}",
        f"- provenance mismatch risk: {result.get('provenance_mismatch_risk')}",
        "",
        "## Warnings",
        *([f"- {item}" for item in result["warnings"]] if result["warnings"] else ["- none"]),
    ]


def compare_runs(a: Path, b: Path) -> dict[str, Any]:
    ia = inspect_run_artifacts(a)
    ib = inspect_run_artifacts(b)
    ma = ia.get("metrics", {})
    mb = ib.get("metrics", {})
    f1a = as_float(ma.get("pair_f1"), 0.0) or 0.0
    f1b = as_float(mb.get("pair_f1"), 0.0) or 0.0
    winner = "a" if f1a > f1b else "b" if f1b > f1a else "tie"
    delta = f1a - f1b
    likely = []
    if ia["config_flags"].get("pairrefine") != ib["config_flags"].get("pairrefine"):
        likely.append("pairrefine differs")
    if ia["config_flags"].get("use_nussinov") != ib["config_flags"].get("use_nussinov"):
        likely.append("decode constraint differs")
    risks = []
    if ia["warnings"]:
        risks.append("run A warnings")
    if ib["warnings"]:
        risks.append("run B warnings")
    return {
        "a": str(a),
        "b": str(b),
        "a_metrics": ma,
        "b_metrics": mb,
        "a_pair_ratio": ia.get("pair_count_ratio"),
        "b_pair_ratio": ib.get("pair_count_ratio"),
        "a_pairrefine": ia["config_flags"].get("pairrefine"),
        "b_pairrefine": ib["config_flags"].get("pairrefine"),
        "a_loss_curve": ia.get("loss_curve"),
        "b_loss_curve": ib.get("loss_curve"),
        "a_gap": ia.get("pair_logit_gap"),
        "b_gap": ib.get("pair_logit_gap"),
        "a_rankAcc": ia.get("rankAcc"),
        "b_rankAcc": ib.get("rankAcc"),
        "winner": winner,
        "main_delta": delta,
        "likely_reason": likely,
        "risk_flags": risks,
    }


def compare_markdown(result: dict[str, Any]) -> list[str]:
    return [
        "# Run Comparison",
        "",
        f"- A: {result['a']}",
        f"- B: {result['b']}",
        f"- winner: {result['winner']}",
        f"- main_delta Pair F1 (A - B): {result['main_delta']}",
        f"- likely reason: {', '.join(result['likely_reason']) if result['likely_reason'] else 'not determined'}",
        f"- risk flags: {', '.join(result['risk_flags']) if result['risk_flags'] else 'none'}",
        "",
        "## Metrics",
        f"- A Pair F1: {result['a_metrics'].get('pair_f1')}",
        f"- B Pair F1: {result['b_metrics'].get('pair_f1')}",
        f"- A precision/recall: {result['a_metrics'].get('pair_precision')} / {result['a_metrics'].get('pair_recall')}",
        f"- B precision/recall: {result['b_metrics'].get('pair_precision')} / {result['b_metrics'].get('pair_recall')}",
        f"- A pair ratio: {result.get('a_pair_ratio')}",
        f"- B pair ratio: {result.get('b_pair_ratio')}",
    ]


def row_pairs(row: dict[str, Any], key: str) -> list[tuple[int, int]]:
    pairs = row.get(key) or []
    cleaned = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            cleaned.append((int(pair[0]), int(pair[1])))
    return cleaned


def case_analysis(pred_path: Path, top_bad: int = 20, top_good: int = 20) -> dict[str, Any]:
    rows = read_jsonl(pred_path)
    missing = []
    required = ["seq", "true_struct", "pred_struct"]
    if rows:
        for field in required:
            if field not in rows[0]:
                missing.append(field)
    if not pred_path.exists():
        return {"pred": str(pred_path), "exists": False, "missing_fields": required, "rows": 0, "cases": []}
    cases = []
    for row in rows:
        seq = row.get("seq", "")
        true = row.get("true_struct", row.get("struct", ""))
        pred = row.get("pred_struct", "")
        try:
            true_pairs = parse_dot_bracket(true) if true else row_pairs(row, "true_pairs")
        except ValueError:
            true_pairs = row_pairs(row, "true_pairs")
        try:
            pred_pairs = parse_dot_bracket(pred) if pred else row_pairs(row, "pred_pairs")
        except ValueError:
            pred_pairs = row_pairs(row, "pred_pairs")
        true_set = {tuple(sorted(pair)) for pair in true_pairs}
        pred_set = {tuple(sorted(pair)) for pair in pred_pairs}
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        valid = bool(row.get("valid", validate_structure(seq, pred) if seq and pred else False))
        ratio = len(pred_set) / len(true_set) if true_set else 0.0
        if not valid:
            reason = "invalid_structure"
        elif pred and set(pred) <= {"."}:
            reason = "all_dot"
        elif len(seq) > 300 and f1 < 0.2:
            reason = "long_sequence_failure"
        elif len(seq) < 50 and f1 < 0.2:
            reason = "short_sequence_failure"
        elif ratio < 0.5:
            reason = "under_pairing"
        elif ratio > 1.5:
            reason = "over_pairing"
        elif recall < 0.3:
            reason = "low_recall"
        elif precision < 0.3:
            reason = "low_precision"
        else:
            reason = "good_case"
        cases.append({
            "id": row.get("id", ""),
            "seq_len": len(seq),
            "true_pair_count": len(true_set),
            "pred_pair_count": len(pred_set),
            "sample_pair_f1": f1,
            "sample_precision": precision,
            "sample_recall": recall,
            "valid": valid,
            "reason": reason,
            "seq": seq,
            "true_struct": true,
            "pred_struct": pred,
        })
    bad = sorted(cases, key=lambda item: item["sample_pair_f1"])[:top_bad]
    good = sorted(cases, key=lambda item: item["sample_pair_f1"], reverse=True)[:top_good]
    reasons = Counter(item["reason"] for item in cases)
    return {
        "pred": str(pred_path),
        "exists": pred_path.exists(),
        "rows": len(rows),
        "missing_fields": missing,
        "reason_counts": reasons,
        "avg_pair_f1": statistics.mean([item["sample_pair_f1"] for item in cases]) if cases else 0.0,
        "bad_cases": bad,
        "good_cases": good,
    }


def case_markdown(result: dict[str, Any]) -> list[str]:
    return [
        "# Case Analysis",
        "",
        f"- predictions: {result['pred']}",
        f"- exists: {result['exists']}",
        f"- rows: {result['rows']}",
        f"- missing fields: {result.get('missing_fields', [])}",
        f"- avg sample Pair F1: {result.get('avg_pair_f1')}",
        "",
        "## Reason Counts",
        *[f"- {key}: {value}" for key, value in dict(result.get("reason_counts", {})).items()],
    ]


def dataset_summary(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(path, limit=max_rows)
    lengths = [int(row.get("length") or len(row.get("seq", ""))) for row in rows]
    families = Counter(str(row.get("family") or "OTHER") for row in rows)
    all_dot = 0
    invalid = 0
    canonical = []
    pair_counts = []
    bad_examples = []
    for row in rows:
        seq = str(row.get("seq", "")).upper().replace("T", "U")
        struct = str(row.get("struct", ""))
        if set(struct) <= {"."}:
            all_dot += 1
        try:
            pairs = parse_dot_bracket(struct)
        except ValueError as exc:
            invalid += 1
            if len(bad_examples) < 10:
                bad_examples.append({"id": row.get("id"), "reason": str(exc)})
            pairs = []
        pair_counts.append(len(pairs))
        if pairs:
            ok = sum(1 for i, j in pairs if i < len(seq) and j < len(seq) and canonical_pair(seq[i], seq[j]))
            canonical.append(ok / len(pairs))
        if struct and len(seq) == len(struct) and not validate_structure(seq, struct):
            invalid += 1
    return {
        "input": str(path),
        "samples": len(rows),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": statistics.mean(lengths) if lengths else 0.0,
        "length_p50": statistics.median(lengths) if lengths else 0.0,
        "family_top20": families.most_common(20),
        "all_dot_ratio": all_dot / len(rows) if rows else 0.0,
        "invalid_count": invalid,
        "avg_pair_count": statistics.mean(pair_counts) if pair_counts else 0.0,
        "canonical_pair_ratio": statistics.mean(canonical) if canonical else 0.0,
        "bad_examples": bad_examples,
    }


class RNAAnalysisAgent:
    """LLM-backed analysis agent that never participates in model inference."""

    def __init__(self, dry_run: bool = False, model: str | None = None, timeout: int = 60, max_retries: int = 2) -> None:
        self.dry_run = dry_run
        self.model_override = model
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)

    def call(self, user_prompt: str) -> str:
        text, _usage = self.call_with_usage(user_prompt)
        return text

    def call_with_usage(self, user_prompt: str) -> tuple[str, dict[str, Any]]:
        if self.dry_run:
            return "# Dry Run Prompt\n\n" + user_prompt, {}
        settings = self._settings()
        payload = {
            "model": settings["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
        }
        request = urllib.request.Request(
            settings["base_url"] + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + settings["api_key"],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error = None
        for _ in range(max(1, self.max_retries + 1)):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"LLM request failed with HTTP {exc.code}: {body[:500]}"
            except urllib.error.URLError as exc:
                last_error = f"LLM request failed: {exc.reason}"
        else:
            raise SystemExit(last_error or "LLM request failed.")
        try:
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            return data["choices"][0]["message"]["content"].strip(), usage if isinstance(usage, dict) else {}
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit("LLM response did not match OpenAI-compatible chat/completions format.") from exc

    def build_diagnose_prompt(self, run_dir: Path) -> tuple[dict[str, Any], str]:
        summary = compact_run(run_dir)
        prompt = f"""Role: Failure Diagnoser.

Diagnose this RNA-OmniDiffusion run.

Metrics JSON:
{json.dumps(summary, indent=2)}

Return:
1. overall health
2. likely failure modes
3. whether pair precision, recall, pair-count ratio, gap, and rankAcc are consistent
4. safe next experiment suggestions
5. what must not be claimed in a paper

Do not compare against external literature benchmarks unless supplied in the JSON.
Do not call the run unhealthy solely because it is below an external threshold.
Use only internal comparisons visible in the JSON, such as random baseline, valid rate, pair-count ratio, gap, and rankAcc.
"""
        return {"summary": summary, "dry_run": self.dry_run}, prompt

    def diagnose(self, run_dir: Path) -> tuple[dict[str, Any], str]:
        data, prompt = self.build_diagnose_prompt(run_dir)
        return data, self.call(prompt)

    def build_schedule_prompt(self, run_dir: Path, config_path: Path | None = None) -> tuple[dict[str, Any], str]:
        summary = compact_run(run_dir)
        config_text = read_text(config_path, max_chars=12000) if config_path else ""
        prompt = f"""Role: Experiment Scheduler.

Create a conservative experiment schedule for RNA-OmniDiffusion.

Current run summary:
{json.dumps(summary, indent=2)}

Current config:
```yaml
{config_text}
```

Rules:
- Do not modify labels, predictions, or benchmark metrics.
- Do not introduce LLM inference, RNA-FM, RNA 3D, ligand, or protein tasks.
- Keep strict Nussinov as the final benchmark path.
- Use only existing CLI patterns:
  - python main.py train --config <config> --device cuda
  - python scripts/eval.py bench --config <config> --ckpt <ckpt> --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
  - python scripts/eval.py bench --config <config> --ckpt <ckpt> --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
  - python scripts/run.py external --configs <configs> --dataset <name> --split random --device cuda --decode nussinov --bench_workers 8 --tag <tag>
- Existing configs you may reference as executable commands: config/candidate.yaml, config/candidate_norefine.yaml, config/candidate_oldmask.yaml, config/oldbase.yaml, config/fixed.yaml, config/scan.json.
- If a new config is needed, write "manual config patch required" and give a YAML patch. Do not put nonexistent config paths such as config/exp1.yaml in runnable commands.
- Prefer 3 to 6 experiments with commands and stop criteria.

Return a schedule table and a fail-fast checklist.
"""
        return {"summary": summary, "config": str(config_path) if config_path else None, "dry_run": self.dry_run}, prompt

    def schedule(self, run_dir: Path, config_path: Path | None = None) -> tuple[dict[str, Any], str]:
        data, prompt = self.build_schedule_prompt(run_dir, config_path)
        return data, self.call(prompt)

    def build_report_prompt(self, inputs: list[Path], max_chars: int = 20000) -> tuple[dict[str, Any], str]:
        sources = {str(path): read_text(path, max_chars=max_chars) for path in inputs}
        prompt = f"""Role: Report Generator.

Draft a paper-ready report section for RNA-OmniDiffusion.

Use only the supplied source text. Do not invent new results.
Keep supported claims separate from unsupported claims.

Sources:
{json.dumps(sources, indent=2)}

Return:
1. concise model summary
2. main results paragraph
3. ablation paragraph
4. limitations paragraph
5. reviewer-risk checklist
"""
        return {"inputs": [str(path) for path in inputs], "dry_run": self.dry_run}, prompt

    def report(self, inputs: list[Path], max_chars: int = 20000) -> tuple[dict[str, Any], str]:
        data, prompt = self.build_report_prompt(inputs, max_chars=max_chars)
        return data, self.call(prompt)

    def build_auditdata_prompt(self, inputs: list[Path], max_rows: int | None = None) -> tuple[dict[str, Any], str]:
        summaries = [dataset_summary(path, max_rows=max_rows) for path in inputs]
        prompt = f"""Role: Data Quality Auditor.

Audit RNA JSONL dataset summaries for paper-quality risks.

Summaries:
{json.dumps(summaries, indent=2)}

Return:
1. data health assessment
2. leakage or split risks visible from the summary
3. structure-label risks
4. what extra checks should be run before paper submission
5. concise action list
"""
        return {"summaries": summaries, "dry_run": self.dry_run}, prompt

    def auditdata(self, inputs: list[Path], max_rows: int | None = None) -> tuple[dict[str, Any], str]:
        data, prompt = self.build_auditdata_prompt(inputs, max_rows=max_rows)
        return data, self.call(prompt)

    def build_inspect_prompt(self, result: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prompt = f"""Role: Training and Run Inspector.

Review this rule-based run inspection. Do not invent additional metrics.

Inspection JSON:
{json.dumps(result, indent=2)}

Return:
1. run health
2. training health
3. pair-head health
4. benchmark consistency
5. recommended next action
"""
        return {"inspection": result, "dry_run": self.dry_run}, prompt

    def build_trace_prompt(self, result: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prompt = f"""Role: Inference Provenance Auditor.

Review this benchmark provenance trace. Do not invent files or rerun benchmark.

Trace JSON:
{json.dumps(result, indent=2)}

Return:
1. official path consistency
2. wrong checkpoint / wrong split / wrong decode risks
3. provenance mismatch risk
4. concise action list
"""
        return {"trace": result, "dry_run": self.dry_run}, prompt

    def build_compare_prompt(self, result: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prompt = f"""Role: Run Comparator.

Compare these two RNA-OmniDiffusion runs using only the JSON.

Comparison JSON:
{json.dumps(result, indent=2)}

Return:
1. winner
2. main_delta
3. likely reason
4. risk flags
5. whether the comparison is paper-table ready
"""
        return {"comparison": result, "dry_run": self.dry_run}, prompt

    def build_case_prompt(self, result: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prompt = f"""Role: Sample Failure Analyst.

Analyze sample-level prediction failures from this case summary.

Case JSON:
{json.dumps(result, indent=2)}

Return:
1. dominant failure modes
2. bad-case patterns
3. good-case patterns
4. whether failures suggest under-pairing, over-pairing, invalidity, or length sensitivity
5. recommended next diagnostic
"""
        return {"case_summary": result, "dry_run": self.dry_run}, prompt

    def build_doctor_prompt(self, result: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prompt = f"""Role: RNA-OmniDiffusion Doctor.

Provide a final read-only diagnosis from this combined report.

Doctor JSON:
{json.dumps(result, indent=2)}

Return:
1. run health
2. training health
3. inference provenance
4. benchmark consistency
5. sample failure modes
6. recommended next action
7. should_run_more_experiments: yes/no
8. should_change_model: yes/no
9. should_update_paper_table: yes/no
"""
        return {"doctor": result, "dry_run": self.dry_run}, prompt

    def _settings(self) -> dict[str, Any]:
        env = load_env()
        api_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
        base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = self.model_override or env.get("LLM_MODEL") or env.get("OPENAI_MODEL") or "gpt-4.1-mini"
        temperature = float(env.get("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(env.get("LLM_MAX_TOKENS", "1200"))
        if not api_key or api_key == "YOUR_KEY":
            raise SystemExit("Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY in .env.")
        if not model or model == "YOUR_MODEL_ID":
            raise SystemExit("Missing LLM model. Set LLM_MODEL in .env.")
        return {
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
