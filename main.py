from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from models.infer import run_infer
from models.train import run_eval, run_smoke, run_train
from models.training import (
    LengthGroupedBatchSampler,
    apply_ablation_settings,
    averages,
    build_datasets_and_tokenizer,
    build_model,
    collect_pair_diagnostics,
    create_tiny_jsonl_dataset,
    decode_batch_tokens,
    deep_update,
    ensure_dataset_paths,
    estimate_loss_options,
    evaluate_model,
    finalize_pair_diagnostics,
    format_epoch_metrics,
    forward_model,
    get_dataset_lengths,
    load_checkpoint,
    load_config,
    loss_from_batch,
    make_loader,
    move_batch_to_device,
    normalize_config,
    print_pair_batch_debug,
    resolve_device,
    run_eval,
    run_infer,
    run_smoke,
    run_train,
    save_checkpoint,
    set_seed,
    synthetic_samples,
    train_model,
    update_running,
    warn_if_collapsed,
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
    config = load_config(args.config) if args.config else {}
    sections = [args.section] if args.section != "all" else ["training", "model", "decoding", "ablation", "agent"]
    result = {section: dict(_section_rows(config, section)) for section in sections}
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print("Adjustable RNA-OmniDiffusion parameters")
    print(f"config: {args.config}")
    for section in sections:
        print(f"\n[{section}]")
        for name, value in _section_rows(config, section):
            print(f"  {name}: {value}")


def run_agent_help(args: argparse.Namespace) -> None:
    print("Agent shell entrypoint:")
    print("  python scripts/llm.py agent --dry_run")
    print()
    print("Common commands:")
    print("  agent> inspect outputs/candidate")
    print("  agent> train candidate")
    print("  agent> run train candidate")
    print("  agent> set target pair_f1 >= 0.75 max_trials 3")
    print("  agent> /memory")
    print()
    print("Safety:")
    print("  Agent is optional and read-only by default.")
    print("  Candidate training requires explicit confirmation.")
    print("  Benchmark execution remains blocked by default.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RNA-OmniDiffusion command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train from a YAML config")
    train.add_argument("--config", default="config/base.yaml", help="YAML training config")
    train.add_argument("--resume", help="Optional checkpoint to resume from")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    train.add_argument("--max_steps", type=int, help="Optional short-run step cap for sanity checks")
    train.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate validation split from a checkpoint")
    eval_parser.add_argument("--config", default="config/base.yaml", help="YAML evaluation config")
    eval_parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    eval_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device")
    eval_parser.set_defaults(func=run_eval)

    infer = subparsers.add_parser("infer", help="Run single-sample inference")
    infer.add_argument("--config", default="config/base.yaml", help="YAML inference config")
    infer.add_argument("--ckpt", required=True, help="Checkpoint path")
    infer.add_argument("--task", required=True, choices=["seq2struct", "invfold"], help="Inference task")
    infer.add_argument("--seq", help="RNA sequence for seq2struct")
    infer.add_argument("--struct", help="Dot-bracket structure for invfold")
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    infer.set_defaults(func=run_infer)

    params = subparsers.add_parser("params", help="Show adjustable config and Agent parameters")
    params.add_argument("--config", default="config/candidate.yaml", help="Config to inspect")
    params.add_argument("--section", default="all", choices=["all", *PARAM_SECTIONS.keys()], help="Parameter section")
    params.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    params.set_defaults(func=run_params)

    agent = subparsers.add_parser("agent", help="Show optional Agent shell usage")
    agent.set_defaults(func=run_agent_help)

    smoke = subparsers.add_parser("smoke", help="Run tiny CPU smoke test")
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)



if __name__ == "__main__":
    main()
