from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from models.display import (
    agent_help_text,
    overview_text,
    params_header,
    params_section,
)


PARAM_SECTIONS = {
    "training": [
        "output_dir",
        "epochs",
        "batch_size",
        "lr",
        "seed",
        "lambda_pair",
        "lambda_struct",
        "lambda_seq",
        "pair_positive_weight",
        "pair_negative_ratio",
        "lambdaConflict",
        "conflictMargin",
        "amp",
        "grad_clip",
    ],
    "model": [
        "hidden_size",
        "num_layers",
        "dropout",
        "pairhead",
        "pairhidden",
        "pairdrop",
        "distbias",
        "pairrefine",
        "pairrefinechannels",
        "pairrefineblocks",
    ],
    "decoding": [
        "decode_source",
        "use_nussinov",
        "pair_threshold",
        "nussinov_gamma",
        "method",
        "min_loop_length",
    ],
    "ablation": [
        "use_pair_head",
        "use_pair_loss",
        "use_nussinov",
        "use_pair_aware_masking",
        "use_motif_span_masking",
        "use_motif_condition",
        "use_family_condition",
    ],
    "agent": [
        "mode",
        "max_api_calls",
        "max_tokens_total",
        "api_timeout",
        "command_timeout",
        "train_timeout",
        "memory",
        "target_tuning",
    ],
}


def _section_rows(config: dict[str, Any], section: str) -> list[tuple[str, Any]]:
    if section == "agent":
        return [(name, "runtime option") for name in PARAM_SECTIONS[section]]
    values = config.get(section, {}) if isinstance(config.get(section), dict) else {}
    keys = PARAM_SECTIONS[section]
    return [(name, values.get(name, "<not set>")) for name in keys]


def run_params(args: argparse.Namespace) -> None:
    # Lazy import to avoid torch dependency for params-only usage
    from models.training import load_config

    config = load_config(args.config) if args.config else {}
    sections = (
        [args.section]
        if args.section != "all"
        else ["training", "model", "decoding", "ablation", "agent"]
    )

    if args.json:
        result = {
            section: dict(_section_rows(config, section)) for section in sections
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    params_header(args.config)
    for section_name in sections:
        items = _section_rows(config, section_name)
        if items:
            params_section(section_name, items)

    # Show quick-start commands at the end
    print()
    print("[commands]")
    print(f"  train : python main.py train --config {args.config} --device cuda")
    print(f"  smoke : python main.py smoke")
    print(f"  agent : python scripts/llm.py agent --dry_run")


def run_agent_help(args: argparse.Namespace) -> None:
    print(agent_help_text())


def run_overview(args: argparse.Namespace) -> None:
    print(overview_text())


# ── Lazy wrappers for torch-dependent commands ──────────────


def _run_train(args: argparse.Namespace) -> None:
    from models.train import run_train
    return run_train(args)


def _run_eval(args: argparse.Namespace) -> None:
    from models.train import run_eval
    return run_eval(args)


def _run_smoke(args: argparse.Namespace) -> None:
    from models.train import run_smoke
    return run_smoke(args)


def _run_infer(args: argparse.Namespace) -> None:
    from models.infer import run_infer
    return run_infer(args)


# ── Parser builder ──────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RNA-OmniDiffusion command line interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train from a YAML config")
    train.add_argument(
        "--config", default="config/candidate.yaml", help="YAML training config"
    )
    train.add_argument("--resume", help="Optional checkpoint to resume from")
    train.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    train.add_argument(
        "--max_steps",
        type=int,
        help="Optional short-run step cap for sanity checks",
    )
    train.set_defaults(func=_run_train)

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate validation split from a checkpoint"
    )
    eval_parser.add_argument("--config", default="config/base.yaml", help="YAML evaluation config")
    eval_parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    eval_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device",
    )
    eval_parser.set_defaults(func=_run_eval)

    infer = subparsers.add_parser("infer", help="Run single-sample inference")
    infer.add_argument("--config", default="config/base.yaml", help="YAML inference config")
    infer.add_argument("--ckpt", required=True, help="Checkpoint path")
    infer.add_argument(
        "--task",
        required=True,
        choices=["seq2struct", "invfold"],
        help="Inference task",
    )
    infer.add_argument("--seq", help="RNA sequence for seq2struct")
    infer.add_argument("--struct", help="Dot-bracket structure for invfold")
    infer.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    infer.set_defaults(func=_run_infer)

    smoke = subparsers.add_parser("smoke", help="Run tiny CPU smoke test")
    smoke.set_defaults(func=_run_smoke)

    params = subparsers.add_parser(
        "params", help="Show adjustable config and Agent parameters"
    )
    params.add_argument(
        "--config", default="config/candidate.yaml", help="Config to inspect"
    )
    params.add_argument(
        "--section",
        default="all",
        choices=["all", *PARAM_SECTIONS.keys()],
        help="Parameter section",
    )
    params.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    params.set_defaults(func=run_params)

    agent = subparsers.add_parser("agent", help="Show optional Agent shell usage")
    agent.set_defaults(func=run_agent_help)

    overview = subparsers.add_parser("overview", help="Show framework overview")
    overview.set_defaults(func=run_overview)

    models = subparsers.add_parser(
        "models", help="Show framework overview (alias for overview)"
    )
    models.set_defaults(func=run_overview)

    return parser
