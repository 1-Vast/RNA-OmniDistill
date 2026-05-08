"""Extract LLM-guided semantic tokens for RNA-OmniDistill-LLM.

Modes:
  --dry_run    Generate prompts only, no API call.
  --fake       Generate deterministic fake annotations (no API).
  --from_jsonl Validate and convert existing LLM annotation JSONL.
  --call_api   Call actual LLM API (requires LLM_API_KEY in env).

The LLM provides weak biological priors only — it does NOT predict
dot-bracket structures, base-pair positions, or pseudo labels.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Direct import to avoid triggering models.teacher.__init__ (which needs numpy)
import importlib.util as _importlib_util
_spec = _importlib_util.spec_from_file_location(
    "llm_semantic", ROOT / "models" / "teacher" / "llm_semantic.py"
)
_llm_semantic = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_llm_semantic)

LLM_SEMANTIC_TOKEN_VOCAB = _llm_semantic.LLM_SEMANTIC_TOKEN_VOCAB
validate_llm_semantic_record = _llm_semantic.validate_llm_semantic_record
semantic_record_to_tokens = _llm_semantic.semantic_record_to_tokens
make_fake_semantic_record = _llm_semantic.make_fake_semantic_record

# ─── Prompt template ───
SYSTEM_PROMPT = (
    "You are a computational RNA biology expert. "
    "Your task is to derive weak biological semantic priors from RNA sequences. "
    "You must not predict exact secondary structures, dot-bracket strings, or base-pair positions."
)

USER_PROMPT_TEMPLATE = """Analyze the RNA sequence below using exactly three reasoning hops.

RNA sequence:
{sequence}

Return a strict JSON object with this schema:
{{
  "length_category": "short|medium|long|very_long",
  "gc_level": "low|medium|high",
  "au_level": "low|medium|high",
  "local_complementarity": "weak|moderate|strong",
  "palindromic_tendency": "low|medium|high",
  "pairing_density_prior": "low|medium|high",
  "stem_loop_tendency": "low|medium|high",
  "hairpin_tendency": "low|medium|high",
  "long_range_pairing_likelihood": "low|medium|high",
  "stability_prior": "low|medium|high",
  "motif_hints": ["stem_loop|hairpin|internal_loop|bulge|multi_loop|unknown"],
  "family_hint": "tRNA_like|rRNA_like|riboswitch_like|miRNA_like|lncRNA_like|unknown",
  "reasoning_hops": [
    "Hop 1: sequence composition and local signal reasoning.",
    "Hop 2: structural tendency reasoning.",
    "Hop 3: biological prior and uncertainty reasoning."
  ],
  "confidence": 0.0
}}

Rules:
- Return JSON only.
- Use exactly 3 reasoning_hops.
- Do not output dot-bracket notation.
- Do not output base-pair positions.
- Do not output pseudo labels.
- If evidence is weak, choose unknown or low confidence.
- Keep each reasoning hop under 30 words."""

# ─── Argument parsing ───
def main():
    parser = argparse.ArgumentParser(description="Extract LLM semantic tokens")
    parser.add_argument("--input", help="Input JSONL with sequences")
    parser.add_argument("--output", default="outputs/llm_semantic_dryrun")
    parser.add_argument("--dry_run", action="store_true", help="Generate prompts only, no API call")
    parser.add_argument("--fake", action="store_true", help="Generate deterministic fake annotations")
    parser.add_argument("--from_jsonl", help="Validate existing LLM annotation JSONL")
    parser.add_argument("--call_api", action="store_true", help="Call actual LLM API (requires env vars)")
    parser.add_argument("--limit", type=int, default=None, help="Max sequences to process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on validation errors")
    parser.add_argument("--no_structure_leak_check", action="store_true")
    parser.add_argument("--provider", default="openai_compatible", help="LLM provider")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args)
    elif args.fake:
        run_fake(args)
    elif args.from_jsonl:
        run_validate(args)
    elif args.call_api:
        run_call_api(args)
    else:
        print("Error: specify --dry_run, --fake, --from_jsonl, or --call_api", file=sys.stderr)
        sys.exit(1)

# ─── Dry run: generate prompts ───
def run_dry_run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sequences = _read_sequences(args)
    if not sequences:
        print("No sequences found.")
        return
    
    prompt_file = out_dir / "llm_prompts.jsonl"
    with open(prompt_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(sequences):
            seq = sample.get("seq", sample.get("sequence", ""))
            user_prompt = USER_PROMPT_TEMPLATE.format(sequence=seq)
            record = {
                "index": i,
                "id": sample.get("id", f"sample_{i}"),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "sequence_length": len(seq),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(sequences)} prompts -> {prompt_file}")

# ─── Fake: generate deterministic annotations ───
def run_fake(args):
    out_dir = Path(args.output).parent if args.output else Path("dataset/semantic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "llm_semantic_fake.jsonl"
    
    if out_path.exists() and not args.overwrite:
        print(f"Error: {out_path} exists. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(1)
    
    sequences = _read_sequences(args)
    if not sequences:
        print("No sequences found. Using synthetic test sequences.")
        sequences = [{"id": f"synth_{i}", "seq": s} for i, s in enumerate([
            "GCAUAGC", "GGGAAACCC", "AUGCAU", "GCUAAGC",
            "CUUGACGAUCAUAGAGCGUUGGAACCACCUGAUCCCUUCCCGAACUCAGAAGUGAAACGACGCAUCGCCGAUGGUAGUGUGGGGUUUCCCCAUGUGAGAGUAGGUCAUCGUCAAGC",
        ])]
    
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in sequences:
            seq = sample.get("seq", sample.get("sequence", ""))
            if not seq:
                continue
            record = make_fake_semantic_record(seq, seed=args.seed + count)
            record["sequence_id"] = sample.get("id", f"sample_{count}")
            errors = validate_llm_semantic_record(record)
            if errors:
                print(f"Warning: validation errors for {record['sequence_id']}: {errors}")
                if args.strict:
                    sys.exit(1)
            tokens = semantic_record_to_tokens(record)
            output = {
                "sequence_id": record["sequence_id"],
                "sequence": seq,
                "semantic_record": record,
                "semantic_tokens": tokens,
                "source_model": "fake_deterministic",
                "prompt_version": "v1.0",
                "created_at": datetime.now().isoformat(),
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    
    print(f"Generated {count} fake semantic annotations -> {out_path}")

# ─── Validate existing annotations ───
def run_validate(args):
    path = Path(args.from_jsonl)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    
    valid = 0
    invalid = 0
    leakage = 0
    
    for i, line in enumerate(lines):
        record = json.loads(line)
        sr = record.get("semantic_record", record)
        errors = validate_llm_semantic_record(sr)
        if errors:
            invalid += 1
            print(f"Line {i+1}: validation errors: {errors}")
            if args.strict:
                sys.exit(1)
        else:
            valid += 1
    
    # Structure leakage check
    if not args.no_structure_leak_check:
        for i, line in enumerate(lines):
            record = json.loads(line)
            sr = record.get("semantic_record", record)
            text = json.dumps(sr).lower()
            if any(c in text for c in ["((", "))", "...", "...."]):
                leakage += 1
                print(f"Line {i+1}: possible structure leakage detected")
    
    print(json.dumps({
        "total": len(lines), "valid": valid, "invalid": invalid,
        "leakage_suspects": leakage, "valid_rate": round(valid / max(1, len(lines)), 3),
    }, indent=2))

# ─── Call API (stub - requires env vars) ───
def run_call_api(args):
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    
    if not api_key:
        print("Error: LLM_API_KEY not set in environment.", file=sys.stderr)
        print("Set LLM_API_KEY and LLM_BASE_URL to call the LLM API.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Would call API at {base_url} (provider={args.provider})")
    print("API call not implemented in this version. Use --dry_run or --fake for now.")
    sys.exit(1)

# ─── Helper ───
def _read_sequences(args):
    if not args.input:
        # Use synthetic sequences if no input
        return []
    
    path = Path(args.input)
    if not path.exists():
        return []
    
    sequences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "seq" in record or "sequence" in record:
                sequences.append(record)
            if args.limit and len(sequences) >= args.limit:
                break
    
    return sequences

if __name__ == "__main__":
    main()
