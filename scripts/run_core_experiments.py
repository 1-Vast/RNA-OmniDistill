from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


CORE_ABLATIONS = ["full", "no_pair_head", "no_nussinov", "random_mask_only"]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def make_quick_config(source: Path, target: Path, epochs: int, output_dir: str | None = None) -> Path:
    config = read_yaml(source)
    config["training"]["epochs"] = epochs
    config["training"]["log_every"] = 1
    config["training"]["warmup_steps"] = 1
    config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
    if output_dir:
        config["training"]["output_dir"] = output_dir
    write_yaml(target, config)
    return target


def command_to_text(command: list[str]) -> str:
    return " ".join(command)


class Runner:
    def __init__(self, experiment: str, dry_run: bool) -> None:
        self.experiment = experiment
        self.dry_run = dry_run
        self.root = Path("outputs") / "core_runs" / experiment
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.commands_path = self.root / "commands.txt"
        self.summary = {"experiment": experiment, "steps": []}

    def run(self, name: str, command: list[str]) -> None:
        text = command_to_text(command)
        with self.commands_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text)
        step = {"name": name, "command": text, "status": "dry_run" if self.dry_run else "pending"}
        if self.dry_run:
            self.summary["steps"].append(step)
            return
        stdout_path = self.logs / f"{len(self.summary['steps']) + 1:02d}_{name}.stdout.log"
        stderr_path = self.logs / f"{len(self.summary['steps']) + 1:02d}_{name}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
        step.update({"stdout": str(stdout_path), "stderr": str(stderr_path), "returncode": proc.returncode})
        if proc.returncode != 0:
            step["status"] = "failed"
            self.summary["steps"].append(step)
            self.write_summary()
            raise SystemExit(f"Step failed: {name}. Command: {text}. See logs: {stdout_path}, {stderr_path}")
        step["status"] = "ok"
        self.summary["steps"].append(step)

    def write_summary(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "run_summary.json"
        path.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        print(f"run summary -> {path}")


def archiveii_full(runner: Runner, quick: bool, skip_download: bool, epochs: int) -> None:
    config = make_quick_config(
        Path("config/config_archiveii.yaml"),
        runner.root / "config_archiveii_used.yaml",
        3 if quick else epochs,
        "outputs/archiveii_full",
    )
    if not skip_download:
        runner.run("download_archiveii", [sys.executable, "scripts/download_rna_datasets.py", "--dataset", "archiveii", "--out", "dataset/raw/archiveii"])
    runner.run("prepare_archiveii", [sys.executable, "scripts/prepare_rna_dataset.py", "--input", "dataset/raw/archiveii", "--output", "dataset/processed_archiveii/clean.jsonl", "--format", "auto", "--max_length", "512"])
    runner.run("check_archiveii", [sys.executable, "scripts/check_dataset.py", "--input", "dataset/processed_archiveii/clean.jsonl", "--output", "dataset/processed_archiveii/clean.checked.jsonl", "--max_length", "512"])
    runner.run("split_archiveii", [sys.executable, "scripts/make_splits.py", "--input", "dataset/processed_archiveii/clean.checked.jsonl", "--out_dir", "dataset/processed_archiveii", "--mode", "random"])
    runner.run("realdata_smoke", [sys.executable, "scripts/run_realdata_smoke.py", "--config", str(config), "--num_train", "32" if quick else "128", "--num_val", "16" if quick else "32", "--steps", "5" if quick else "100"])
    runner.run("overfit_tiny", [sys.executable, "scripts/overfit_tiny.py", "--config", str(config), "--num_samples", "8", "--steps", "20" if quick else "200"])
    train_cmd = [sys.executable, "main.py", "train", "--config", str(config)]
    if quick:
        train_cmd += ["--train_subset", "32", "--max_steps", "6"]
    runner.run("train", train_cmd)
    runner.run("benchmark", [sys.executable, "scripts/run_benchmark.py", "--config", str(config), "--ckpt", "outputs/archiveii_full/best.pt", "--split", "test"])
    runner.run("analyze", [sys.executable, "scripts/analyze_training.py", "--log", "outputs/archiveii_full/train_log.jsonl", "--out", "outputs/archiveii_full/training_analysis.json"])
    runner.run("diagnose", [sys.executable, "scripts/diagnose_predictions.py", "--pred", "outputs/archiveii_full/predictions_test.jsonl", "--out", "outputs/archiveii_full/prediction_diagnosis.json"])


def archiveii_core_ablation(runner: Runner, quick: bool, epochs: int) -> None:
    cmd = [sys.executable, "scripts/run_ablations.py", "--base_config", "config/config_archiveii.yaml", "--only", *CORE_ABLATIONS, "--epochs", str(3 if quick else epochs)]
    if quick:
        cmd.append("--quick")
    runner.run("archiveii_core_ablation", cmd)


def rnastralign512_full(runner: Runner, quick: bool, epochs: int) -> None:
    config = make_quick_config(
        Path("config/config_rnastralign512.yaml"),
        runner.root / "config_rnastralign512_used.yaml",
        3 if quick else epochs,
        "outputs/rnastralign512_full",
    )
    required = [Path("dataset/processed_rnastralign512/train.jsonl"), Path("dataset/processed_rnastralign512/val.jsonl"), Path("dataset/processed_rnastralign512/test.jsonl")]
    if not all(path.exists() for path in required) and not runner.dry_run:
        raise SystemExit("RNAStrAlign.512 processed splits are missing. Prepare them before running rnastralign512_full.")
    train_cmd = [sys.executable, "main.py", "train", "--config", str(config)]
    if quick:
        train_cmd += ["--train_subset", "32", "--max_steps", "6"]
    runner.run("train_rnastralign512", train_cmd)
    runner.run("benchmark_rnastralign512", [sys.executable, "scripts/run_benchmark.py", "--config", str(config), "--ckpt", "outputs/rnastralign512_full/best.pt", "--split", "test"])


def rnastralign_to_archiveii(runner: Runner) -> None:
    external = Path("dataset/processed_archiveii/clean.checked.jsonl")
    ckpt = Path("outputs/rnastralign512_full/best.pt")
    if not runner.dry_run and not external.exists():
        raise SystemExit(f"External ArchiveII JSONL missing: {external}")
    if not runner.dry_run and not ckpt.exists():
        raise SystemExit(f"RNAStrAlign checkpoint missing: {ckpt}")
    runner.run(
        "benchmark_external_archiveii",
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--config",
            "config/config_external_archiveii.yaml",
            "--ckpt",
            str(ckpt),
            "--input_jsonl",
            str(external),
            "--output_prefix",
            "outputs/rnastralign512_full/archiveii_external",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core RNA-OmniDiffusion real-data experiments.")
    parser.add_argument("--experiment", required=True, choices=["archiveii_full", "archiveii_core_ablation", "rnastralign512_full", "rnastralign_to_archiveii", "all_core"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    runner = Runner(args.experiment, args.dry_run)
    if args.experiment in {"archiveii_full", "all_core"}:
        archiveii_full(runner, args.quick, args.skip_download, args.epochs)
    if args.experiment in {"archiveii_core_ablation", "all_core"}:
        archiveii_core_ablation(runner, args.quick, args.epochs)
    if args.experiment in {"rnastralign512_full", "all_core"}:
        rnastralign512_full(runner, args.quick, args.epochs)
    if args.experiment in {"rnastralign_to_archiveii", "all_core"}:
        rnastralign_to_archiveii(runner)
    runner.write_summary()


if __name__ == "__main__":
    main()

