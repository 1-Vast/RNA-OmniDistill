from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_VARIANTS = [
    "full",
    "no_pair_head",
    "no_nussinov",
    "random_mask_only",
    "no_pair_aware_masking",
    "no_motif_span_masking",
    "no_motif_family_condition",
    "token_decode",
    "pair_decode",
    "hybrid_decode",
]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_command(command: list[str], dry_run: bool) -> tuple[bool, str]:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return True, "dry_run"
    proc = subprocess.run(command, text=True)
    if proc.returncode != 0:
        return False, f"exit_code={proc.returncode}"
    return True, "ok"


def ensure_data(config: dict[str, Any]) -> None:
    missing = []
    for key in ["train_jsonl", "val_jsonl", "test_jsonl"]:
        path = Path(config["data"][key])
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise SystemExit(
            "Required split JSONL files are missing: "
            + ", ".join(missing)
            + ". Run prepare_rna_dataset.py and make_splits.py first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RNA-OmniDiffusion ablation variants.")
    parser.add_argument("--base_config", default="config/config.yaml")
    parser.add_argument("--split", default="random")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise SystemExit(f"Base config does not exist: {base_config_path}")
    base_config = load_yaml(base_config_path)
    ensure_data(base_config)

    variants = args.only if args.only else DEFAULT_VARIANTS
    report = {"split": args.split, "quick": args.quick, "variants": []}
    for variant in variants:
        ablation_path = Path("config") / "ablations" / f"{variant}.yaml"
        if not ablation_path.exists():
            message = f"missing ablation config: {ablation_path}"
            report["variants"].append({"variant": variant, "status": "failed", "reason": message})
            if not args.continue_on_error:
                raise SystemExit(message)
            continue

        config = deep_update(json.loads(json.dumps(base_config)), load_yaml(ablation_path))
        config["training"]["epochs"] = 3 if args.quick else args.epochs
        if args.quick:
            config["training"]["log_every"] = 1
            config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
        output_dir = Path(config["training"]["output_dir"])
        config_path = output_dir / "config_ablation.yaml"
        write_yaml(config_path, config)

        commands = [
            [sys.executable, "main.py", "train", "--config", str(config_path)],
            [
                sys.executable,
                "scripts/run_benchmark.py",
                "--config",
                str(config_path),
                "--ckpt",
                str(output_dir / "best.pt"),
                "--split",
                "test",
            ],
            [
                sys.executable,
                "scripts/analyze_training.py",
                "--log",
                str(output_dir / "train_log.jsonl"),
                "--out",
                str(output_dir / "training_analysis.json"),
            ],
            [
                sys.executable,
                "scripts/diagnose_predictions.py",
                "--pred",
                str(output_dir / "predictions_test.jsonl"),
                "--out",
                str(output_dir / "prediction_diagnosis.json"),
            ],
        ]
        if args.quick:
            commands[0].extend(["--train_subset", "32", "--max_steps", "6"])
            commands[1].extend(["--max_samples", "16"])

        status = "ok"
        reason = "ok"
        for command in commands:
            ok, reason = run_command(command, args.dry_run)
            if not ok:
                status = "failed"
                break
        report["variants"].append(
            {
                "variant": variant,
                "status": status,
                "reason": reason,
                "config": str(config_path),
                "output_dir": str(output_dir),
            }
        )
        if status != "ok" and not args.continue_on_error:
            break

    report_path = Path("outputs") / "ablations" / "ablation_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ablation run report -> {report_path}")
    if any(item["status"] != "ok" for item in report["variants"]) and not args.continue_on_error:
        raise SystemExit("One or more ablation variants failed. See ablation_run_report.json.")


if __name__ == "__main__":
    main()
