"""Build training policy JSONL for LLM-guided relational curriculum.

Modes:
  --mode rule   Rule-based deterministic policy (no LLM)
  --mode llm    LLM-driven policy (dry-run by default)
  --mode mock   Mock LLM policy for testing
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def rule_policy(sample, seed=42):
    """Deterministic rule-based curriculum policy."""
    rng = random.Random(hash(sample.get("id", "")) ^ seed)
    length = len(sample.get("seq", ""))
    
    if length <= 80:
        regime = "structured_short_rna"
        mask_mode = "span"
        mask_ratio = round(rng.uniform(0.12, 0.18), 2)
        span_len = rng.choice([2, 3, 4])
        neg_mode = "near_diagonal"
        near_w = round(rng.uniform(0.6, 0.9), 2)
        long_w = round(rng.uniform(0.1, 0.3), 2)
        sampling_weight = round(rng.uniform(1.1, 1.3), 2)
        curriculum_stage = 1
    elif length <= 200:
        regime = "mixed"
        mask_mode = "span"
        mask_ratio = round(rng.uniform(0.15, 0.22), 2)
        span_len = rng.choice([4, 5, 6])
        neg_mode = "canonical_hard_negative"
        near_w = round(rng.uniform(0.2, 0.4), 2)
        long_w = round(rng.uniform(0.4, 0.6), 2)
        sampling_weight = round(rng.uniform(0.9, 1.1), 2)
        curriculum_stage = 2
    else:
        regime = "long_range_rna"
        mask_mode = "span"
        mask_ratio = round(rng.uniform(0.10, 0.15), 2)
        span_len = rng.choice([6, 7, 8])
        neg_mode = "medium_long"
        near_w = round(rng.uniform(0.1, 0.2), 2)
        long_w = round(rng.uniform(0.6, 0.8), 2)
        sampling_weight = round(rng.uniform(1.0, 1.2), 2)
        curriculum_stage = 3
    
    return {
        "family": sample.get("family", "unknown"),
        "source": sample.get("source", "unknown"),
        "regime": regime,
        "mask_policy": {"mode": mask_mode, "mask_ratio": mask_ratio, "span_len": span_len},
        "negative_policy": {"mode": neg_mode, "near_diagonal_weight": near_w, "long_range_weight": long_w},
        "sampling_policy": {"sampling_weight": sampling_weight, "curriculum_stage": curriculum_stage},
        "confidence": round(rng.uniform(0.7, 1.0), 2),
    }

def mock_llm_policy(sample, seed=42):
    """Mock LLM policy - same structure as rule but with different fallback values."""
    base = rule_policy(sample, seed)
    base["confidence"] = round(random.Random(seed).uniform(0.3, 0.7), 2)
    return base

def llm_policy_dryrun(sample, seed=42):
    """Generate LLM prompt without calling API."""
    seq = sample.get("seq", "")
    family = sample.get("family", "unknown")
    length = len(seq)
    gc = sum(1 for c in seq.upper() if c in "GC") / max(1, length)
    
    prompt = f"""You are a training policy compiler for RNA structure prediction.
Given the RNA family metadata below, output a JSON training policy.

Family: {family}
Length: {length} nt
GC content: {gc:.2f}
Source: {sample.get("source", "unknown")}

Output JSON matching this schema:
{{
  "family": "{family}",
  "source": "{sample.get("source", "unknown")}",
  "regime": "local_hairpin|structured_short_rna|long_range_rna|mixed|unknown",
  "mask_policy": {{"mode": "random|span", "mask_ratio": 0.15, "span_len": 5}},
  "negative_policy": {{"mode": "random|near_diagonal|medium_long|canonical_hard_negative", "near_diagonal_weight": 0.5, "long_range_weight": 0.5}},
  "sampling_policy": {{"sampling_weight": 1.0, "curriculum_stage": 2}},
  "confidence": 0.5
}}

Rules:
- JSON only, no prose.
- No dot-bracket or base-pair labels.
- No structure predictions.
- Use the constrained enum values only.
- Set confidence based on how well the metadata supports your choice.
"""
    return {"prompt": prompt, "mock_policy": mock_llm_policy(sample, seed)}

BUILDERS = {"rule": rule_policy, "llm": llm_policy_dryrun, "mock": mock_llm_policy}

def main():
    parser = argparse.ArgumentParser(description="Build training policy JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["rule", "llm", "mock"], default="rule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    builder = BUILDERS[args.mode]
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_path, "w") as f:
        for sample in samples:
            policy = builder(sample, args.seed + count)
            row = {"id": sample.get("id", f"s{count}"), "policy": policy}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    
    print(json.dumps({"mode": args.mode, "count": count, "output": str(out_path)}, indent=2))

if __name__ == "__main__":
    main()
