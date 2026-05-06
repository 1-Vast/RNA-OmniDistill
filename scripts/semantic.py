"""Offline LLM semantic/task annotation for RNA data.

Usage:
  python scripts/semantic.py annotate --input data.jsonl --out semantic.jsonl --provider none --max_samples 64
  python scripts/semantic.py annotate_tasks --input data.jsonl --out task_semantic.jsonl --provider none --max_samples 64
  python scripts/semantic.py audit --input semantic.jsonl --out outputs/audit

Provider=none: rule-based templates, no API call.
Provider=ark/openai/gemini: real LLM API call (requires .env).
"""

from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
from typing import Optional

FAMILY_TYPES = ["tRNA", "riboswitch", "ribozyme", "miRNA", "snRNA", "snoRNA", "rRNA", "tmRNA", "lncRNA", "unknown"]
MOTIF_TYPES = ["hairpin", "stem_loop", "bulge", "internal_loop", "multiloop", "pseudoknot", "unknown"]
BIAS_TYPES = ["stem_rich", "loop_rich", "cloverleaf", "long_range_pairing", "balanced", "unknown"]
FUNCTION_TAGS = ["translation", "catalytic", "regulatory", "splicing", "structural", "defense", "unknown"]
CONSTRAINT_HINTS = ["STEM_RICH", "CONSERVED_LOOP", "LONG_RANGE", "COMPACT", "UNKNOWN"]
TASK_NAMES = ["SEQ2STRUCT", "INVFOLD", "INPAINT", "MOTIF_REPAIR"]
MASK_REGIONS = ["stem", "hairpin_loop", "internal_loop", "bulge", "multiloop", "random"]

def _load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]

def _rule_family_type(family: str, description: str = "") -> str:
    f = (family or "").upper()
    for ft in FAMILY_TYPES[:-1]:
        if ft.upper() in f or ft.upper() in (description or "").upper():
            return ft
    return "unknown"

def _rule_motif(struct: str) -> str:
    if not struct: return "unknown"
    s = struct.replace("[","(").replace("]","(").replace("{","(").replace("}",")")
    has_pair = "(" in s
    has_hairpin = False
    for i in range(len(s)-3):
        if s[i] == "(" and s[i+1:i+4].count(".") >= 2 and ")" in s[i+1:i+5]:
            has_hairpin = True
            break
    if has_hairpin: return "hairpin"
    if has_pair: return "stem_loop"
    return "unknown"

def _rule_bias(struct: str) -> str:
    if not struct: return "unknown"
    pairs = sum(1 for c in struct if c in "([{")
    dots = struct.count(".")
    l = max(len(struct), 1)
    ratio = pairs / l
    if ratio > 0.4: return "stem_rich"
    if dots / l > 0.7: return "loop_rich"
    return "unknown"

def _rule_function_tag(family: str, description: str = "") -> str:
    f = (family or "").upper()
    d = (description or "").upper()
    if any(k in f + d for k in ["TRNA", "TRANSLAT", "RIBOSOM"]): return "translation"
    if any(k in f + d for k in ["RIBOZYM", "CATALYT", "SELF_CLEAV"]): return "catalytic"
    if any(k in f + d for k in ["RIBOSWITCH", "REGULAT", "SENSOR"]): return "regulatory"
    if any(k in f + d for k in ["SPLIC", "INTRON", "SNRNA"]): return "splicing"
    return "unknown"

def _rule_constraint(struct: str) -> str:
    if not struct: return "UNKNOWN"
    pairs = sum(1 for c in struct if c in "([{")
    l = max(len(struct), 1)
    if pairs / l > 0.45: return "STEM_RICH"
    return "UNKNOWN"

def _rule_task_hint(struct: str) -> list:
    hints = ["SEQ2STRUCT"]
    if "(" in struct and "." in struct: hints.append("INVFOLD")
    return hints[:2]

def _rule_mask_region(struct: str) -> str:
    if not struct: return "random"
    has_pair = "(" in struct or "[" in struct
    has_dots = "." * 4 in struct
    if has_pair: return "stem"
    if has_dots: return "hairpin_loop"
    return "random"

def _rule_repair_target(struct: str) -> str:
    return "conserved_loop" if "." * 3 in (struct or "") else "unknown"

def _rule_difficulty(length: int) -> str:
    if length > 300: return "hard"
    if length > 150: return "medium"
    return "easy"

def _llm_call(provider: str, prompt: str, env_vars: dict) -> Optional[str]:
    if provider == "ark":
        import requests
        resp = requests.post(
            f"{env_vars.get('LLM_BASE_URL','')}/chat/completions",
            headers={"Authorization": f"Bearer {env_vars.get('LLM_API_KEY','')}", "Content-Type": "application/json"},
            json={"model": env_vars.get("LLM_MODEL",""), "messages": [{"role":"user","content":prompt}], "temperature": float(env_vars.get("LLM_TEMPERATURE", 0.1)), "max_tokens": int(env_vars.get("LLM_MAX_TOKENS", 512))},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return None
    elif provider in ["openai", "gemini"]:
        import requests
        base = "https://api.openai.com/v1" if provider == "openai" else "https://generativelanguage.googleapis.com/v1beta"
        resp = requests.post(
            f"{base}/chat/completions" if provider == "openai" else f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={env_vars.get('LLM_API_KEY','')}",
            headers={"Authorization": f"Bearer {env_vars.get('LLM_API_KEY','')}", "Content-Type": "application/json"},
            json={"model": env_vars.get("LLM_MODEL","gpt-4o-mini"), "messages": [{"role":"user","content":prompt}], "temperature": float(env_vars.get("LLM_TEMPERATURE", 0.1))},
            timeout=60
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"] if provider == "openai" else data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        return None
    return None

def _load_env(env_path: str = ".env") -> dict:
    env = {}
    if Path(env_path).exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("\"'")
    return env

def cmd_annotate(args):
    input_path = args.input
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider = args.provider or "none"
    max_samples = int(args.max_samples or 0)
    resume = args.resume

    records = _load_jsonl(input_path)
    if max_samples > 0:
        records = records[:max_samples]

    existing = {}
    if resume and out_path.exists():
        for r in _load_jsonl(str(out_path)):
            existing[r.get("id", "")] = r

    env = {}
    if provider != "none":
        env_path = args.env or ".env"
        env = _load_env(env_path)
        if not env.get("LLM_API_KEY"):
            print(f"WARNING: No LLM_API_KEY in {env_path}. Falling back to provider=none.")
            provider = "none"

    count = 0
    cache = {}
    with open(out_path, "w") as fout:
        for rec in records:
            rid = rec.get("id", str(count))
            if rid in existing:
                fout.write(json.dumps(existing[rid]) + "\n")
                count += 1
                continue

            family = rec.get("family", "unknown")
            description = rec.get("description", rec.get("metadata", {}).get("description", ""))
            seq = rec.get("seq", "")
            struct = rec.get("struct", "")

            if provider == "none":
                semantic = {
                    "family_type": _rule_family_type(family, description),
                    "primary_motif": _rule_motif(struct),
                    "structure_bias": _rule_bias(struct),
                    "function_tag": _rule_function_tag(family, description),
                    "constraint_hint": _rule_constraint(struct),
                }
                source = "rule"
                model_name = "rule-based"
                prompt_hash = "rule-000000000000"
            else:
                prompt = f"""RNA family: {family}
Description: {description}
Length: {len(seq)}
Structure preview: {struct[:50]}...

Classify into standard ontology. Output JSON only:
{{"family_type": "one of {','.join(FAMILY_TYPES)}", "primary_motif": "one of {','.join(MOTIF_TYPES)}", "structure_bias": "one of {','.join(BIAS_TYPES)}", "function_tag": "one of {','.join(FUNCTION_TAGS)}", "constraint_hint": "one of {','.join(CONSTRAINT_HINTS)}"}}"""
                prompt_hash = _hash_prompt(prompt)
                if prompt_hash in cache:
                    semantic = cache[prompt_hash]
                else:
                    raw = _llm_call(provider, prompt, env)
                    if raw:
                        try:
                            semantic = json.loads(raw)
                            cache[prompt_hash] = semantic
                        except json.JSONDecodeError:
                            semantic = {"family_type": "unknown", "primary_motif": "unknown", "structure_bias": "unknown", "function_tag": "unknown", "constraint_hint": "UNKNOWN"}
                    else:
                        semantic = {"family_type": "unknown", "primary_motif": "unknown", "structure_bias": "unknown", "function_tag": "unknown", "constraint_hint": "UNKNOWN"}
                source = "llm"
                model_name = env.get("LLM_MODEL", provider)
                time.sleep(0.5)

            out_rec = dict(rec)
            out_rec["semantic"] = semantic
            out_rec["semantic_source"] = source
            out_rec["semantic_model"] = model_name
            out_rec["semantic_prompt_hash"] = prompt_hash
            fout.write(json.dumps(out_rec) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"  Annotated {count}/{len(records)}")

    print(f"Annotated {count} records -> {out_path}")

def cmd_annotate_tasks(args):
    input_path = args.input
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider = args.provider or "none"
    max_samples = int(args.max_samples or 0)
    resume = args.resume

    records = _load_jsonl(input_path)
    if max_samples > 0:
        records = records[:max_samples]

    existing = {}
    if resume and out_path.exists():
        for r in _load_jsonl(str(out_path)):
            existing[r.get("id", "")] = r

    env = {}
    if provider != "none":
        env_path = args.env or ".env"
        env = _load_env(env_path)
        if not env.get("LLM_API_KEY"):
            print(f"WARNING: No LLM_API_KEY. Falling back to provider=none.")
            provider = "none"

    count = 0
    with open(out_path, "w") as fout:
        for rec in records:
            rid = rec.get("id", str(count))
            if rid in existing:
                fout.write(json.dumps(existing[rid]) + "\n")
                count += 1
                continue

            family = rec.get("family", "")
            description = rec.get("description", "")
            seq = rec.get("seq", "")
            struct = rec.get("struct", "")

            if provider == "none":
                task_semantic = {
                    "recommended_tasks": _rule_task_hint(struct),
                    "mask_region": _rule_mask_region(struct),
                    "repair_target": _rule_repair_target(struct),
                    "constraint_type": "UNKNOWN",
                    "difficulty": _rule_difficulty(len(seq)),
                }
                source = "rule"
                model_name = "rule-based"
                prompt_hash = "rule-task-00000000"
            else:
                prompt = f"""RNA: family={family}, description={description}, length={len(seq)}
Tasks: {','.join(TASK_NAMES)}
Suggest multitask configuration. Output JSON only:
{{"recommended_tasks": ["task1","task2"], "mask_region": "one of {','.join(MASK_REGIONS)}", "repair_target": "conserved_loop|stem|motif|unknown", "constraint_type": "preserve_stem|preserve_loop|maintain_pairing|unknown", "difficulty": "easy|medium|hard"}}"""
                prompt_hash = _hash_prompt(prompt)
                raw = _llm_call(provider, prompt, env)
                if raw:
                    try: task_semantic = json.loads(raw)
                    except: task_semantic = {"recommended_tasks":["SEQ2STRUCT"],"mask_region":"random","repair_target":"unknown","constraint_type":"UNKNOWN","difficulty":"medium"}
                else:
                    task_semantic = {"recommended_tasks":["SEQ2STRUCT"],"mask_region":"random","repair_target":"unknown","constraint_type":"UNKNOWN","difficulty":"medium"}
                source = "llm"
                model_name = env.get("LLM_MODEL", provider)
                time.sleep(0.5)

            out_rec = dict(rec)
            out_rec["task_semantic"] = task_semantic
            out_rec["semantic_source"] = source
            out_rec["semantic_model"] = model_name
            out_rec["semantic_prompt_hash"] = prompt_hash
            fout.write(json.dumps(out_rec) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"  Task-annotated {count}/{len(records)}")

    print(f"Task-annotated {count} records -> {out_path}")

def cmd_audit(args):
    input_path = args.input
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(input_path)
    if not records:
        print("No records found")
        return

    stats = {
        "total": len(records),
        "invalid_json": 0,
        "missing_semantic": 0,
        "missing_task_semantic": 0,
        "semantic_source": {},
        "family_type_dist": {},
        "primary_motif_dist": {},
        "structure_bias_dist": {},
        "function_tag_dist": {},
        "constraint_hint_dist": {},
        "recommended_tasks_dist": {},
        "mask_region_dist": {},
        "difficulty_dist": {},
        "unknown_ratio": 0.0,
        "duplicate_ids": 0,
    }

    seen_ids = set()
    for r in records:
        rid = r.get("id", "")
        if rid in seen_ids:
            stats["duplicate_ids"] += 1
        seen_ids.add(rid)

        src = r.get("semantic_source", "none")
        stats["semantic_source"][src] = stats["semantic_source"].get(src, 0) + 1

        sem = r.get("semantic", {})
        if not sem:
            stats["missing_semantic"] += 1
        else:
            for field, dist_key in [("family_type","family_type_dist"), ("primary_motif","primary_motif_dist"), ("structure_bias","structure_bias_dist"), ("function_tag","function_tag_dist"), ("constraint_hint","constraint_hint_dist")]:
                val = sem.get(field, "unknown")
                stats[dist_key][val] = stats[dist_key].get(val, 0) + 1

        ts = r.get("task_semantic", {})
        if not ts:
            stats["missing_task_semantic"] += 1
        else:
            for t in ts.get("recommended_tasks", []):
                stats["recommended_tasks_dist"][t] = stats["recommended_tasks_dist"].get(t, 0) + 1
            mr = ts.get("mask_region", "unknown")
            stats["mask_region_dist"][mr] = stats["mask_region_dist"].get(mr, 0) + 1
            diff = ts.get("difficulty", "unknown")
            stats["difficulty_dist"][diff] = stats["difficulty_dist"].get(diff, 0) + 1

    unknown_count = sum(1 for r in records if r.get("semantic",{}).get("family_type") == "unknown")
    stats["unknown_ratio"] = unknown_count / max(len(records), 1)

    with open(out_dir / "audit.json", "w") as f:
        json.dump(stats, f, indent=2)

    md = f"""# Semantic Annotation Audit

- Total records: {stats['total']}
- Missing semantic: {stats['missing_semantic']}
- Missing task_semantic: {stats['missing_task_semantic']}
- Unknown ratio (family_type): {stats['unknown_ratio']:.2%}
- Duplicate IDs: {stats['duplicate_ids']}

## Semantic Source Distribution
| Source | Count |
|---|---|
"""
    for src, cnt in sorted(stats["semantic_source"].items()):
        md += f"| {src} | {cnt} |\n"

    md += "\n## Family Type Distribution\n| Type | Count |\n|---|---|\n"
    for ft, cnt in sorted(stats["family_type_dist"].items(), key=lambda x: -x[1]):
        md += f"| {ft} | {cnt} |\n"

    md += "\n## Motif Distribution\n| Motif | Count |\n|---|---|\n"
    for m, cnt in sorted(stats["primary_motif_dist"].items(), key=lambda x: -x[1]):
        md += f"| {m} | {cnt} |\n"

    with open(out_dir / "audit.md", "w") as f:
        f.write(md)

    print(f"Audit complete: {out_dir}/audit.json, {out_dir}/audit.md")
    print(f"  Records: {stats['total']}, Unknown: {stats['unknown_ratio']:.1%}")

def main():
    parser = argparse.ArgumentParser(description="LLM semantic/task annotation for RNA")
    sub = parser.add_subparsers(dest="command")

    p_ann = sub.add_parser("annotate")
    p_ann.add_argument("--input", required=True)
    p_ann.add_argument("--out", required=True)
    p_ann.add_argument("--provider", default="none")
    p_ann.add_argument("--env", default=".env")
    p_ann.add_argument("--max_samples", type=int, default=0)
    p_ann.add_argument("--resume", action="store_true")

    p_task = sub.add_parser("annotate_tasks")
    p_task.add_argument("--input", required=True)
    p_task.add_argument("--out", required=True)
    p_task.add_argument("--provider", default="none")
    p_task.add_argument("--env", default=".env")
    p_task.add_argument("--max_samples", type=int, default=0)
    p_task.add_argument("--resume", action="store_true")

    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--input", required=True)
    p_audit.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "annotate_tasks":
        cmd_annotate_tasks(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
