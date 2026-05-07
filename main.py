from __future__ import annotations

from typing import Iterable

from models.training import (
    LengthGroupedBatchSampler,
    apply_ablation_settings,
    averages,
    build_datasets_and_tokenizer,
    build_model,
    build_parser,
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
    main as _training_main,
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


def main(argv: Iterable[str] | None = None) -> None:
    _training_main(argv)


if __name__ == "__main__":
    main()
