from __future__ import annotations

from typing import Any


def sep(char: str = "=", width: int = 72) -> None:
    """Print a separator line of repeated characters."""
    print(char * width)


def banner(text: str) -> None:
    """Print a banner with text centered between separator lines."""
    sep()
    print(text.center(72))
    sep()


def section(text: str) -> None:
    """Print a section header with a blank line before."""
    print()
    print(text)


def key_value_table(
    items: list[tuple[str, str]], indent: int = 0, align_colon: bool = True
) -> None:
    """Print aligned key-value pairs.

    When align_colon is True, keys are left-aligned so all colons line up.
    """
    if not items:
        return
    prefix = " " * indent
    if align_colon:
        key_width = max(len(k) for k, _ in items) + 2
        for key, value in items:
            print(f"{prefix}{key:<{key_width}}: {value}")
    else:
        for key, value in items:
            print(f"{prefix}{key}: {value}")


def _format_lr(value: Any) -> str:
    """Format a learning rate value for human-readable display."""
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.0e}"
        s = f"{value:.10f}".rstrip("0").rstrip(".")
        return s
    return str(value)


def train_startup(config: dict, device_str: str, gpu_name: str) -> None:
    """Print a training startup summary banner with config details."""
    training = config.get("training", {})
    model_cfg = config.get("model", {})

    items: list[tuple[str, str]] = [
        ("Config", str(training.get("_config_path", "?"))),
        ("Output", str(training.get("output_dir", "?"))),
        ("Device", device_str),
        ("GPU", gpu_name),
        ("Seed", str(training.get("seed", "?"))),
        ("Epochs", str(training.get("epochs", "?"))),
        ("Batch size", str(training.get("batch_size", "?"))),
        ("LR", _format_lr(training.get("lr", "?"))),
        ("AMP", str(training.get("amp", "?"))),
        ("Pair head", str(model_cfg.get("pairhead", "?"))),
        ("Pair refine", str(model_cfg.get("pairrefine", "?"))),
        ("Decode source", str(config.get("decoding", {}).get("decode_source", "?"))),
        ("Nussinov", str(config.get("decoding", {}).get("use_nussinov", "?"))),
        ("Save best by", str(training.get("save_best_by", "?"))),
    ]
    banner("RNA-OmniDiffusion Training")
    key_value_table(items)
    sep()


def epoch_line(metrics: dict, epoch: int, total_epochs: int) -> str:
    """Return a single compact string for one epoch (does NOT print).

    Fields in order: train_loss, val_loss, val_pair_f1, valid, all_dot, lr, epoch_time.
    Missing keys show ``-`` for that position.
    """
    parts = [f"Epoch {epoch:03d}/{total_epochs:03d}"]

    fields = [
        ("train_loss", "train_loss", "{:.4f}"),
        ("val_loss", "val_loss", "{:.4f}"),
        ("val_pair_f1", "val_pair_f1", "{:.4f}"),
        ("val_valid_structure_rate", "valid", "{:.3f}"),
        ("val_all_dot_ratio", "all_dot", "{:.2f}"),
    ]

    for key, display, fmt in fields:
        if key in metrics and metrics[key] is not None:
            parts.append(f"{display} {fmt.format(metrics[key])}")
        else:
            parts.append(f"{display} -")

    # Learning rate
    lr_val = metrics.get("lr", metrics.get("learning_rate"))
    if lr_val is not None:
        parts.append(f"lr {lr_val:.2e}")
    else:
        parts.append("lr -")

    # Epoch time
    epoch_time = metrics.get("epoch_time", metrics.get("time"))
    if epoch_time is not None:
        parts.append(f"{epoch_time:.1f}s")
    else:
        parts.append("-s")

    return " | ".join(parts)


def checkpoint_saved(
    path: str,
    is_best: bool,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> None:
    """Print a short checkpoint notification with ASCII ``[OK]`` prefix."""
    if is_best and metric_name is not None and metric_value is not None:
        print(f"[OK] best checkpoint saved: {path} ({metric_name}={metric_value:.4f})")
    else:
        print(f"[OK] last checkpoint saved: {path}")


def early_stopping_summary(
    best_epoch: int, best_metric_name: str, best_value: float, patience: int
) -> None:
    """Print early stopping summary with best epoch and metric."""
    print("Early stopping:")
    print(f"  best epoch : {best_epoch}")
    print(f"  best metric: {best_metric_name}={best_value:.4f}")
    print(f"  patience   : {patience}")


def training_complete(
    best_path: str, last_path: str, log_path: str, output_dir: str
) -> None:
    """Print training completion summary with checkpoint paths and next step."""
    section("Training completed")
    print(f"Best checkpoint : {best_path}")
    print(f"Last checkpoint : {last_path}")
    print(f"Train log       : {log_path}")
    print("Next step       : python main.py eval --config <config> --ckpt <checkpoint>")


def inference_header(
    task: str, checkpoint: str, config_path: str, device: str
) -> None:
    """Print inference header using key_value_table."""
    items: list[tuple[str, str]] = [
        ("Task", task),
        ("Checkpoint", checkpoint),
        ("Config", config_path),
        ("Device", device),
    ]
    print("RNA-OmniDiffusion Inference")
    key_value_table(items)


def inference_result_seq2struct(
    input_seq: str, predicted_struct: str, length: int
) -> None:
    """Print seq2struct inference result."""
    print(f"Input seq  : {input_seq}")
    print(f"Predicted  : {predicted_struct}")
    print(f"Length     : {length}")


def inference_result_invfold(
    input_struct: str, predicted_seq: str, length: int
) -> None:
    """Print inverse folding inference result."""
    print(f"Input struct: {input_struct}")
    print(f"Predicted   : {predicted_seq}")
    print(f"Length      : {length}")


def overview_text() -> str:
    """Return the framework overview description string."""
    return (
        "RNA-OmniDistill Framework Overview\n"
        "\n"
        "Core model:\n"
        "  RNAOmniDiffusion\n"
        "    Transformer encoder with task, segment, time, and position embeddings.\n"
        "\n"
        "Heads:\n"
        "  token heads\n"
        "    sequence head: predicts RNA sequence tokens.\n"
        "    structure head: predicts dot-bracket structure tokens.\n"
        "    general head: fallback token prediction.\n"
        "  pair head\n"
        "    predicts base-pair logits over sequence positions.\n"
        "  pair refine\n"
        "    optional 2D refinement over pair logits.\n"
        "\n"
        "Training tasks:\n"
        "  seq2struct\n"
        "    input sequence, predict structure.\n"
        "  invfold\n"
        "    input structure, predict sequence.\n"
        "  inpaint\n"
        "    mask spans, recover sequence/structure tokens.\n"
        "  motif_control\n"
        "    condition on motif/family tokens when enabled.\n"
        "\n"
        "Decoding:\n"
        "  strict Nussinov\n"
        "    converts pair logits into valid non-crossing dot-bracket structures.\n"
        "  token decoding\n"
        "    iterative unmasking over structure tokens.\n"
        "  hybrid decoding\n"
        "    combines token compatibility with pair logits.\n"
        "\n"
        "Commands:\n"
        "  train     Train from a YAML config.\n"
        "  eval      Evaluate validation split from checkpoint.\n"
        "  infer     Run single-sample inference.\n"
        "  smoke     Run tiny sanity test.\n"
        "  params    Inspect adjustable config parameters.\n"
        "  overview  Show this framework overview.\n"
        "  models    Alias for overview.\n"
        "  models    Alias for overview."
    )


def params_header(config_path: str) -> None:
    """Print adjustable-parameters header."""
    print("Adjustable Parameters")
    print(f"Config: {config_path}")


def params_section(
    section_name: str,
    items: list[tuple[str, Any]],
    align_width: int | None = None,
) -> None:
    """Print a parameter section with aligned key-value columns."""
    if align_width is None and items:
        align_width = max(len(k) for k, _ in items)
    elif align_width is None:
        align_width = 0
    print(f"[{section_name}]")
    for key, value in items:
        print(f"  {key:<{align_width}}  {value}")
