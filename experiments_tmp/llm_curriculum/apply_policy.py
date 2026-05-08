"""Apply curriculum policy to sequence JSONL, producing policy-enriched output."""
import argparse, json, random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Sequence JSONL")
    parser.add_argument("--policy", required=True, help="Policy JSONL from build_policy.py")
    parser.add_argument("--output", required=True, help="Output policy-enriched JSONL")
    parser.add_argument("--shuffle_policy", action="store_true", help="Shuffle policies across samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load sequences
    seqs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                seqs.append(json.loads(line))
    
    # Load policies
    policies = {}
    with open(args.policy) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                policies[r["id"]] = r["policy"]
    
    # Shuffle if requested
    if args.shuffle_policy:
        rng = random.Random(args.seed)
        policy_ids = list(policies.keys())
        policy_values = [policies[k] for k in policy_ids]
        rng.shuffle(policy_values)
        policies = dict(zip(policy_ids, policy_values))
    
    # Apply
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    matched = 0
    with open(out_path, "w") as f:
        for sample in seqs:
            sid = sample.get("id", "")
            if sid in policies:
                p = policies[sid]
                sample["curriculum_policy"] = {
                    "mask_mode": p["mask_policy"]["mode"],
                    "mask_ratio": p["mask_policy"]["mask_ratio"],
                    "span_len": p["mask_policy"]["span_len"],
                    "negative_mode": p["negative_policy"]["mode"],
                    "sampling_weight": p["sampling_policy"]["sampling_weight"],
                    "curriculum_stage": p["sampling_policy"]["curriculum_stage"],
                }
                matched += 1
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(json.dumps({
        "total": len(seqs), "matched": matched,
        "shuffled": args.shuffle_policy, "output": str(out_path),
    }, indent=2))

if __name__ == "__main__":
    main()
