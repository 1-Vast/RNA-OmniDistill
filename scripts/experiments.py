"""RNA-OmniDistill experiment management CLI."""
import argparse, json, yaml, csv, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─── plan ───
def run_plan(args):
    """Write a Markdown experiment plan."""
    plan = f"""# RNA-OmniDistill Experiment Plan
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## A. Reproducibility (3-seed repeat)
| Experiment | Seeds | Config |
|---|---|---|
| RNACentral 50k D-RNAFM | 42,43,44,45,46,47 | rnacentral_drnafm_50k_seed*.yaml |

## B. Pretraining Source Comparison
| Experiment | Teacher | Config |
|---|---|---|
| Baseline (no pretrain) | none | candidate.yaml |
| Rfam D-only 50k | none | rfam_donly_50k.yaml |
| Rfam D-RNAFM 50k | RNA-FM | rfam_drnafm_50k.yaml |
| bpRNA D-RNAFM 50k | RNA-FM | bprna_drnafm_50k.yaml |
| RNACentral D-RNAFM 50k | RNA-FM | rnacentral_drnafm_50k.yaml |
| RNACentral D-RNAFM 100k | RNA-FM | rnacentral_drnafm_100k.yaml |

## C. Distillation Ablation
| Experiment | Distill | Config |
|---|---|---|
| RNACentral D-only 50k | no | rnacentral_donly_50k.yaml |
| RNACentral D-RNAFM 50k | yes | rnacentral_drnafm_50k.yaml |

## D. Relation Refinement Ablation
| Experiment | Refine | Config |
|---|---|---|
| No refine | false | norefine.yaml |
| Refine blocks=1 | true | refine_b1.yaml |
| Refine blocks=2 | true | refine_b2.yaml |
| Refine channels=16 | true | refine_c16.yaml |

## E. Decode Strategy
| Decode | Description |
|---|---|
| strict Nussinov | Primary paper metric |
| greedy | Pair-head probe only |
| token-only | Negative result (valid rate ~0) |

## F. External Benchmark
| Dataset | Split |
|---|---|
| ArchiveII | test |
| bpRNA-1m(90) | random test |
| Rfam | family-disjoint (if available) |

## G. Calibration & Error Analysis
- Pair probability calibration curve
- Length-stratified Pair F1
- Family-stratified Pair F1
- False positive/negative pair export

## H. Runtime Profiling
- GPU forward time vs CPU Nussinov time
- Length-bucketed runtime
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan, encoding="utf-8")
    print(f"Plan written to {out}")

# ─── make_configs ───
def run_make_configs(args):
    """Generate experiment configs from a base config."""
    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        print(f"Error: base config not found: {base}", file=sys.stderr)
        sys.exit(1)

    with open(base) as f:
        cfg = yaml.safe_load(f)

    # Generate configs for each experiment
    experiments = {
        "candidate.yaml": {},
        "rnacentral_drnafm_50k_seed42.yaml": {
            "training.output_dir": "outputs/runs/rnacentral_drnafm_50k_seed42",
            "training.init_from_pretrain": "outputs/runs/rnacentral_drnafm_pretrain_50k_seed42/best.pt",
            "training.load_encoder_only": True,
        },
        "norefine.yaml": {
            "model.pairrefine": False,
            "training.output_dir": "outputs/runs/norefine",
        },
        "refine_b2.yaml": {
            "model.pairrefineblocks": 2,
            "training.output_dir": "outputs/runs/refine_b2",
        },
        "refine_b3.yaml": {
            "model.pairrefineblocks": 3,
            "training.output_dir": "outputs/runs/refine_b3",
        },
        "refine_c16.yaml": {
            "model.pairrefinechannels": 16,
            "training.output_dir": "outputs/runs/refine_c16",
        },
    }

    for name, overrides in experiments.items():
        # Deep copy and apply overrides
        import copy
        exp_cfg = copy.deepcopy(cfg)
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            target = exp_cfg
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value

        out_path = out_dir / name
        with open(out_path, "w") as f:
            yaml.dump(exp_cfg, f, sort_keys=False)
        print(f"Wrote {out_path}")

    print(f"Generated {len(experiments)} configs in {out_dir}")

# ─── summarize ───
def run_summarize(args):
    """Summarize experiment runs into a Markdown table."""
    runs = [Path(r) for r in args.runs]

    rows = []
    for run in runs:
        bench_path = run / "benchmark.json"
        if bench_path.exists():
            with open(bench_path) as f:
                bench = json.load(f)
            model = bench.get("overall", {}).get("model", {})
            rows.append({
                "run": run.name,
                "pair_f1": model.get("pair_f1", 0),
                "precision": model.get("pair_precision", 0),
                "recall": model.get("pair_recall", 0),
                "valid": model.get("valid_structure_rate", 0),
            })
        else:
            rows.append({"run": run.name, "pair_f1": "no_benchmark"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["| Run | Pair F1 | Precision | Recall | Valid |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        if isinstance(row["pair_f1"], str):
            lines.append(f"| {row['run']} | {row['pair_f1']} | - | - | - |")
        else:
            lines.append(f"| {row['run']} | {row['pair_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['valid']:.4f} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {out}")

# ─── export_table ───
def run_export_table(args):
    """Export experiment results as CSV."""
    runs = [Path(r) for r in args.runs]

    rows = []
    for run in runs:
        bench_path = run / "benchmark.json"
        if bench_path.exists():
            with open(bench_path) as f:
                bench = json.load(f)
            model = bench.get("overall", {}).get("model", {})
            rows.append({
                "run": run.name,
                "pair_f1": f"{model.get('pair_f1', 0):.4f}",
                "precision": f"{model.get('pair_precision', 0):.4f}",
                "recall": f"{model.get('pair_recall', 0):.4f}",
                "valid": f"{model.get('valid_structure_rate', 0):.4f}",
                "pair_ratio": f"{model.get('pair_ratio', 0):.4f}" if 'pair_ratio' in model else "N/A",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "pair_f1", "precision", "recall", "valid", "pair_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Table exported to {out}")

# ─── main ───
def main():
    parser = argparse.ArgumentParser(description="RNA-OmniDistill experiment manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Write experiment plan document")
    p.add_argument("--out", default="docs/experiment_plan.md")
    p.set_defaults(func=run_plan)

    p = sub.add_parser("make_configs", help="Generate experiment configs from base")
    p.add_argument("--base", required=True)
    p.add_argument("--out", default="config/experiments")
    p.set_defaults(func=run_make_configs)

    p = sub.add_parser("summarize", help="Summarize runs into Markdown table")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_summarize)

    p = sub.add_parser("export_table", help="Export results as CSV")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_export_table)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
