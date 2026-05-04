from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


TOY_ROWS = [
    {"id": "toy_0001", "seq": "GCAUAGC", "struct": "((...))", "family": "toy"},
    {"id": "toy_0002", "seq": "GGGAAACCC", "struct": "(((...)))", "family": "toy"},
    {"id": "toy_0003", "seq": "AUGCAU", "struct": "(....)", "family": "toy"},
    {"id": "toy_0004", "seq": "GCUAAGC", "struct": "((...))", "family": "toy"},
]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_toy_fallback() -> None:
    raw = Path("dataset/processed/archiveii.clean.jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", encoding="utf-8") as handle:
        for row in TOY_ROWS:
            handle.write(json.dumps(row) + "\n")
    checked = Path("dataset/processed/archiveii.checked.jsonl")
    with checked.open("w", encoding="utf-8") as handle:
        for row in TOY_ROWS:
            handle.write(json.dumps(row) + "\n")
    out = Path("dataset/processed_archiveii")
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train.jsonl": TOY_ROWS, "val.jsonl": TOY_ROWS[:2], "test.jsonl": TOY_ROWS[2:]}
    for name, rows in splits.items():
        with (out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


class SuiteRunner:
    def __init__(self, dataset: str, mode: str, dry_run: bool) -> None:
        self.dataset = dataset
        self.mode = mode
        self.dry_run = dry_run
        self.root = Path("outputs") / "potential" / f"{dataset}_{mode}"
        self.core_root = Path("outputs") / "core_runs" / f"{dataset}_{mode}"
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.core_root.mkdir(parents=True, exist_ok=True)
        self.commands_path = self.root / "commands.txt"
        self.core_commands_path = self.core_root / "commands.txt"
        if self.commands_path.exists():
            self.commands_path.unlink()
        if self.core_commands_path.exists():
            self.core_commands_path.unlink()
        self.summary = {
            "dataset": dataset,
            "mode": mode,
            "dry_run": dry_run,
            "toy_fallback_only": False,
            "steps": [],
        }

    def run(self, name: str, command: list[str], allow_quick_fallback: bool = False) -> bool:
        text = " ".join(command)
        with self.commands_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        with self.core_commands_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text)
        step = {"name": name, "command": text, "status": "dry_run" if self.dry_run else "pending"}
        if self.dry_run:
            self.summary["steps"].append(step)
            self.write_summary()
            return True
        stdout_path = self.logs / f"{len(self.summary['steps']) + 1:02d}_{name}.stdout.log"
        stderr_path = self.logs / f"{len(self.summary['steps']) + 1:02d}_{name}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
        step.update({"stdout": str(stdout_path), "stderr": str(stderr_path), "returncode": proc.returncode})
        if proc.returncode != 0:
            step["status"] = "failed"
            self.summary["failed_step"] = name
            self.summary["failed_command"] = text
            self.summary["log_path"] = str(stderr_path)
            self.summary["steps"].append(step)
            if allow_quick_fallback and self.mode == "quick":
                print(f"Warning: {name} failed; using toy fallback only for pipeline validation. See {stderr_path}")
                self.summary["toy_fallback_only"] = True
                create_toy_fallback()
                return False
            self.write_summary()
            raise SystemExit(f"Step failed: {name}. Command: {text}. See logs: {stdout_path}, {stderr_path}")
        step["status"] = "ok"
        self.summary["steps"].append(step)
        self.write_summary()
        return True

    def write_summary(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        self.core_root.mkdir(parents=True, exist_ok=True)
        (self.core_root / "run_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(data: dict, method: str, key: str) -> float:
    return float(data.get("overall", {}).get(method, {}).get(key, 0.0))


def write_full_decision_summary(output_dir: Path, run_summary_path: Path, benchmark_path: Path) -> None:
    model = read_json(output_dir / "model_potential.json")
    agent = read_json(output_dir / "agent_potential.json")
    benchmark = read_json(benchmark_path)
    run_summary = read_json(run_summary_path)
    metrics = dict(model.get("metrics", {}))
    if not metrics:
        avg_true = metric(benchmark, "model", "avg_true_pair_count")
        metrics = {
            "test_pair_precision": metric(benchmark, "model", "pair_precision"),
            "test_pair_recall": metric(benchmark, "model", "pair_recall"),
            "test_pair_f1": metric(benchmark, "model", "pair_f1"),
            "random_valid_pair_baseline_f1": metric(benchmark, "random_pair", "pair_f1"),
            "all_dot_baseline_f1": metric(benchmark, "all_dot", "pair_f1"),
            "all_dot_ratio": metric(benchmark, "model", "all_dot_ratio"),
            "valid_structure_rate": metric(benchmark, "model", "valid_structure_rate"),
            "canonical_pair_ratio": metric(benchmark, "model", "canonical_pair_ratio"),
            "avg_pred_pair_count": metric(benchmark, "model", "avg_pred_pair_count"),
            "avg_true_pair_count": avg_true,
            "pair_count_ratio": metric(benchmark, "model", "avg_pred_pair_count") / max(1e-8, avg_true),
            "positive_pair_logit_mean": 0.0,
            "negative_pair_logit_mean": 0.0,
            "pair_logit_gap": 0.0,
        }
    checks = {
        "pair_f1_beats_random_baseline": metrics.get("test_pair_f1", 0.0) > metrics.get("random_valid_pair_baseline_f1", 0.0),
        "positive_logit_exceeds_negative": metrics.get("positive_pair_logit_mean", 0.0) > metrics.get("negative_pair_logit_mean", 0.0),
        "all_dot_ratio_below_0_3": metrics.get("all_dot_ratio", 1.0) < 0.3,
        "pair_count_ratio_in_0_5_to_1_5": 0.5 <= metrics.get("pair_count_ratio", 0.0) <= 1.5,
        "valid_structure_rate_near_1": metrics.get("valid_structure_rate", 0.0) >= 0.95,
        "canonical_pair_ratio_near_1": metrics.get("canonical_pair_ratio", 0.0) >= 0.95,
    }
    full_success = bool(run_summary.get("steps")) and not run_summary.get("failed_step") and not run_summary.get("toy_fallback_only")
    recommendation = "Run archiveii_core_ablation next." if model.get("potential") in {"High", "Medium"} else "Do not run ablation yet; diagnose pair head, pair labels, and Nussinov score first."
    if metrics.get("pair_logit_gap", 0.0) <= 0:
        recommendation += " Pair logit gap is non-positive; prioritize checking pair head positive/negative ordering."
    result = {
        "archiveii_full_success": full_success,
        "model_potential": model.get("potential", "Low"),
        "agent_potential": agent.get("agent_potential", "Low"),
        "metrics": metrics,
        "checks": checks,
        "next_step_recommendation": recommendation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_decision_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ArchiveII Full Decision Summary",
        "",
        f"ArchiveII full completed: **{full_success}**",
        f"Model potential: **{result['model_potential']}**",
        f"Agent-readiness: **{result['agent_potential']}**",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "test_pair_precision",
        "test_pair_recall",
        "test_pair_f1",
        "random_valid_pair_baseline_f1",
        "all_dot_baseline_f1",
        "all_dot_ratio",
        "valid_structure_rate",
        "canonical_pair_ratio",
        "avg_pred_pair_count",
        "avg_true_pair_count",
        "pair_count_ratio",
        "positive_pair_logit_mean",
        "negative_pair_logit_mean",
        "pair_logit_gap",
    ]:
        value = metrics.get(key, 0.0)
        lines.append(f"| {key} | {float(value):.4f} |")
    lines.extend(["", "## Key Judgments", ""])
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Step", "", recommendation])
    (output_dir / "full_decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_config(mode: str) -> Path:
    source = Path("config/config_archiveii.yaml")
    config = load_yaml(source)
    if mode == "quick":
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 2
        config["training"]["output_dir"] = "outputs/archiveii_full_quick"
        config["training"]["warmup_steps"] = 1
        config["training"]["log_every"] = 1
        config["model"]["hidden_size"] = 64
        config["model"]["num_layers"] = 2
        config["model"]["num_heads"] = 4
        config["model"]["max_position_embeddings"] = 512
        config["data"]["max_length"] = min(int(config["data"].get("max_length", 512)), 128)
        config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
        path = Path("outputs/potential/archiveii_quick/config_archiveii_quick.yaml")
    else:
        config["training"]["output_dir"] = "outputs/archiveii_full"
        path = Path("outputs/potential/archiveii_full/config_archiveii_full.yaml")
    write_yaml(path, config)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run potential evaluation suite without any Agent framework.")
    parser.add_argument("--dataset", choices=["archiveii"], default="archiveii")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runner = SuiteRunner(args.dataset, args.mode, args.dry_run)
    if args.skip_download and not Path("dataset/raw/archiveii").exists():
        raise SystemExit(
            "dataset/raw/archiveii does not exist. Please run: "
            "python scripts/download_rna_datasets.py --dataset archiveii --out dataset/raw/archiveii"
        )
    config_path = make_config(args.mode)
    output_dir = "outputs/archiveii_full_quick" if args.mode == "quick" else "outputs/archiveii_full"
    smoke_steps = "50" if args.mode == "quick" else "100"
    overfit_steps = "50" if args.mode == "quick" else "200"

    try:
        if not args.skip_download:
            ok = runner.run("download_archiveii", [sys.executable, "scripts/download_rna_datasets.py", "--dataset", "archiveii", "--out", "dataset/raw/archiveii"], allow_quick_fallback=args.mode == "quick")
            if not ok and runner.mode == "quick":
                pass
        if not runner.summary["toy_fallback_only"]:
            ok = runner.run("prepare", [sys.executable, "scripts/prepare_rna_dataset.py", "--input", "dataset/raw/archiveii", "--output", "dataset/processed/archiveii.clean.jsonl", "--format", "auto", "--max_length", "512"], allow_quick_fallback=args.mode == "quick")
            if not ok and runner.mode == "quick":
                pass
        if not runner.summary["toy_fallback_only"]:
            runner.run("check", [sys.executable, "scripts/check_dataset.py", "--input", "dataset/processed/archiveii.clean.jsonl", "--output", "dataset/processed/archiveii.checked.jsonl", "--max_length", "512"], allow_quick_fallback=args.mode == "quick")
            runner.run("split", [sys.executable, "scripts/make_splits.py", "--input", "dataset/processed/archiveii.checked.jsonl", "--out_dir", "dataset/processed_archiveii", "--mode", "random"], allow_quick_fallback=args.mode == "quick")

        runner.run("realdata_smoke", [sys.executable, "scripts/run_realdata_smoke.py", "--config", str(config_path), "--num_train", "128", "--num_val", "32", "--steps", smoke_steps])
        runner.run("overfit_tiny", [sys.executable, "scripts/overfit_tiny.py", "--config", str(config_path), "--num_samples", "8", "--steps", overfit_steps])
        train_cmd = [sys.executable, "main.py", "train", "--config", str(config_path)]
        if args.mode == "quick":
            train_cmd += ["--train_subset", "32", "--max_steps", "8"]
        runner.run("train_full", train_cmd)
        benchmark_cmd = [sys.executable, "scripts/run_benchmark.py", "--config", str(config_path), "--ckpt", f"{output_dir}/best.pt", "--split", "test"]
        if args.mode == "quick":
            benchmark_cmd += ["--max_samples", "16"]
        runner.run("benchmark", benchmark_cmd)
        runner.run("analyze", [sys.executable, "scripts/analyze_training.py", "--log", f"{output_dir}/train_log.jsonl", "--out", f"{output_dir}/training_analysis.json"])
        runner.run("diagnose", [sys.executable, "scripts/diagnose_predictions.py", "--pred", f"{output_dir}/predictions_test.jsonl", "--out", f"{output_dir}/prediction_diagnosis.json"])

        ablation_summary = "outputs/ablations/archiveii_core_summary/summary.json"
        if args.mode == "quick":
            ablations = ["full", "no_nussinov"]
            ablation_cmd = [sys.executable, "scripts/run_ablations.py", "--base_config", str(config_path), "--only", *ablations]
            ablation_cmd.append("--quick")
            runner.run("core_ablation", ablation_cmd)
            ablation_inputs = [f"outputs/ablations/{name}/benchmark_test.json" for name in ablations]
            runner.run(
                "compare_core_ablation",
                [
                    sys.executable,
                    "scripts/compare_ablations.py",
                    "--inputs",
                    *ablation_inputs,
                    "--out",
                    "outputs/ablations/archiveii_core_summary",
                ],
            )
        else:
            ablation_summary = str(runner.root / "no_ablation_summary.json")
            if not args.dry_run:
                Path(ablation_summary).write_text(json.dumps({"rows": [], "note": "ArchiveII full run only; core ablation intentionally skipped."}, indent=2) + "\n", encoding="utf-8")
        model_out = str(runner.root)
        runner.run(
            "summarize_model_potential",
            [
                sys.executable,
                "scripts/summarize_model_potential.py",
                "--benchmark",
                f"{output_dir}/benchmark_test.json",
                "--training_analysis",
                f"{output_dir}/training_analysis.json",
                "--prediction_diagnosis",
                f"{output_dir}/prediction_diagnosis.json",
                "--ablation_summary",
                ablation_summary,
                "--out",
                model_out,
            ],
        )
        runner.run(
            "evaluate_agent_potential",
            [
                sys.executable,
                "scripts/evaluate_agent_potential.py",
                "--model_potential",
                f"{model_out}/model_potential.json",
                "--training_analysis",
                f"{output_dir}/training_analysis.json",
                "--prediction_diagnosis",
                f"{output_dir}/prediction_diagnosis.json",
                "--ablation_summary",
                ablation_summary,
                "--run_summary",
                str(runner.core_root / "run_summary.json"),
                "--out",
                model_out,
            ],
        )
        if args.mode == "full":
            write_full_decision_summary(
                runner.root,
                runner.core_root / "run_summary.json",
                Path(output_dir) / "benchmark_test.json",
            )
    finally:
        if runner.summary["toy_fallback_only"]:
            runner.summary["warning"] = "toy fallback only; not valid for model potential conclusion"
        runner.write_summary()
        print(f"run summary -> {runner.root / 'run_summary.json'}")


if __name__ == "__main__":
    main()
