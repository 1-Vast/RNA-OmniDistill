"""Generate experiment YAML configs for RNA-OmniDistill experiments."""
import argparse
import os
from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "config"


def make_donly_pretrain(train_jsonl, val_jsonl, seed, epochs, output_dir):
    """D-only pretrain config: no teacher."""
    config = {
        "data": {
            "train_jsonl": train_jsonl,
            "val_jsonl": val_jsonl,
            "allow_unlabeled": True,
            "max_length": 512,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "length_grouping": True,
            "length_bucket_size": 50,
        },
        "tasks": {"seq_denoise": 1.0},
        "model": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "dropout": 0.1,
            "max_position_embeddings": 2048,
            "use_distill_head": False,
            "teacher_dim": 640,
            "distill_pool": "mean",
        },
        "training": {
            "batch_size": 64,
            "epochs": epochs,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "lambda_seq": 1.0,
            "lambda_struct": 0.0,
            "lambda_pair": 0.0,
            "lambda_distill": 0.0,
            "grad_clip": 1.0,
            "amp": True,
            "seed": seed,
            "output_dir": output_dir,
            "save_best_by": "val_loss",
            "early_stopping_patience": 3,
            "min_delta": 0.0001,
            "log_every": 50,
            "train_decode_structures": False,
            "val_decode_samples": 0,
            "val_max_batches": 4,
        },
        "decoding": {"use_nussinov": False, "decode_source": "token"},
        "ablation": {
            "use_pair_head": False,
            "use_pair_loss": False,
            "use_nussinov": False,
            "use_pair_aware_masking": False,
            "use_motif_span_masking": False,
            "use_motif_condition": False,
            "use_family_condition": False,
        },
    }
    return config


def make_drnafm_pretrain(train_jsonl, val_jsonl, seed, epochs, output_dir):
    """D-RNAFM pretrain config: with teacher distillation."""
    config = make_donly_pretrain(train_jsonl, val_jsonl, seed, epochs, output_dir)
    config["model"]["use_distill_head"] = True
    config["training"]["lambda_distill"] = 0.05
    return config


def make_finetune(pretrain_dir, output_dir, seed, epochs=20):
    """Fine-tune config: encoder-only from pretrain."""
    config = {
        "data": {
            "train_jsonl": "dataset/archive/train.jsonl",
            "val_jsonl": "dataset/archive/val.jsonl",
            "test_jsonl": "dataset/archive/test.jsonl",
            "max_length": 512,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "length_grouping": True,
            "length_bucket_size": 50,
        },
        "tasks": {
            "seq2struct": 0.35,
            "invfold": 0.25,
            "inpaint": 0.25,
            "motif_control": 0.15,
        },
        "model": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "dropout": 0.1,
            "max_position_embeddings": 2048,
            "distbias": True,
            "pairrefine": True,
        },
        "training": {
            "batch_size": 24,
            "epochs": epochs,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "lambda_pair": 5.0,
            "lambda_struct": 1.0,
            "lambda_seq": 1.0,
            "pair_negative_ratio": 5,
            "pair_positive_weight": 10.0,
            "structure_positive_weight": 1.0,
            "grad_clip": 1.0,
            "amp": True,
            "seed": seed,
            "output_dir": output_dir,
            "init_from_pretrain": f"{pretrain_dir}/best.pt",
            "load_encoder_only": True,
            "save_best_by": "val_loss",
            "early_stopping_patience": 10,
            "min_delta": 0.0001,
            "log_every": 20,
            "train_decode_structures": False,
            "val_decode_samples": 16,
            "val_max_batches": 4,
        },
        "decoding": {
            "num_steps": 32,
            "min_loop_length": 3,
            "use_nussinov": True,
            "allow_wobble": True,
            "decode_source": "pair",
            "pair_threshold": 0.25,
            "nussinov_gamma": 2.0,
            "topk_pairs": 20,
        },
        "ablation": {
            "use_pair_head": True,
            "use_pair_loss": True,
            "use_nussinov": True,
            "use_pair_aware_masking": False,
            "use_motif_span_masking": False,
            "use_motif_condition": False,
            "use_family_condition": False,
        },
        "baselines": {
            "run_all_dot": True,
            "run_random_pair": True,
            "run_rnafold": False,
            "rnafold_bin": "RNAfold",
        },
        "debug": {"check_pair_batch": False},
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate RNA-OmniDistill experiment configs"
    )
    parser.add_argument("--name", required=True, help="Experiment name (e.g., rfam50k, bprna50k)")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--teacher_train_jsonl", default="")
    parser.add_argument("--teacher_val_jsonl", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs_pretrain", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    parser.add_argument("--output_prefix", default="outputs")
    args = parser.parse_args()

    prefix = args.output_prefix
    seed = args.seed

    # 1) D-only pretrain
    d_only_name = f"seq_pretrain_{args.name}_seed{seed}"
    d_only_config = make_donly_pretrain(
        args.train_jsonl,
        args.val_jsonl,
        seed,
        args.epochs_pretrain,
        f"{prefix}/{d_only_name}",
    )
    out_path = Path(f"config/{d_only_name}.yaml")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(d_only_config, f, sort_keys=False)
    print(f"Wrote {out_path}")

    # 2) D-only fine-tune
    d_only_ft_name = f"candidate_from_seq_pretrain_{args.name}_seed{seed}"
    d_only_ft_config = make_finetune(
        f"{prefix}/{d_only_name}",
        f"{prefix}/{d_only_ft_name}",
        seed,
        args.epochs_finetune,
    )
    out_path = Path(f"config/{d_only_ft_name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(d_only_ft_config, f, sort_keys=False)
    print(f"Wrote {out_path}")

    # 3) D-RNAFM pretrain (only if teacher data provided)
    if args.teacher_train_jsonl and args.teacher_val_jsonl:
        drnafm_name = f"seq_pretrain_rnafm_{args.name}_seed{seed}"
        drnafm_config = make_drnafm_pretrain(
            args.teacher_train_jsonl,
            args.teacher_val_jsonl,
            seed,
            args.epochs_pretrain,
            f"{prefix}/{drnafm_name}",
        )
        out_path = Path(f"config/{drnafm_name}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(drnafm_config, f, sort_keys=False)
        print(f"Wrote {out_path}")

        # 4) D-RNAFM fine-tune
        drnafm_ft_name = f"candidate_from_rnafm_pretrain_{args.name}_seed{seed}"
        drnafm_ft_config = make_finetune(
            f"{prefix}/{drnafm_name}",
            f"{prefix}/{drnafm_ft_name}",
            seed,
            args.epochs_finetune,
        )
        out_path = Path(f"config/{drnafm_ft_name}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(drnafm_ft_config, f, sort_keys=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
