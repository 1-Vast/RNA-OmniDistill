from __future__ import annotations

import json
import os
import statistics
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from utils.struct import canonical_pair, parse_dot_bracket, validate_structure


ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You are an RNA secondary-structure experiment analysis agent.
You analyze logs, metrics, dataset summaries, and paper text.
You must not modify labels, benchmark metrics, predictions, or test data.
You must not participate in benchmark inference.
You must not claim LLMs improve RNA-OmniDiffusion model performance.
You must not invent files, commands, datasets, metrics, or baselines.
Use only repository commands with main.py, scripts/eval.py, scripts/run.py, scripts/data.py, scripts/audit.py, scripts/probe.py, and scripts/llm.py.
Do not cite external performance thresholds unless they are supplied in the user prompt.
Treat every output as experiment assistance, not as model output.
Be concise, technical, and explicit about uncertainty."""


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value
    return env


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def read_text(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def write_report(out: Path, stem: str, data: dict[str, Any], markdown: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if data.get("dry_run"):
        (out / "request.json").write_text(json.dumps({**data, "prompt": markdown}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "prompt.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {out / 'request.json'}")
        print(f"Wrote {out / 'prompt.md'}")
        return
    (out / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{stem}.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out / (stem + '.json')}")
    print(f"Wrote {out / (stem + '.md')}")


def compact_run(run_dir: Path) -> dict[str, Any]:
    benchmark = read_json(run_dir / "benchmark.json")
    analysis = read_json(run_dir / "analysis.json")
    diagnosis = read_json(run_dir / "diagnosis.json")
    train_rows = read_jsonl(run_dir / "trainlog.jsonl")
    last_train = train_rows[-1] if train_rows else {}
    model = benchmark.get("overall", {}).get("model", {})
    random_base = benchmark.get("overall", {}).get("random", {})
    avg_pred = float(model.get("avg_pred_pair_count") or 0.0)
    avg_true = float(model.get("avg_true_pair_count") or 0.0)
    return {
        "run_dir": str(run_dir),
        "pair_f1": model.get("pair_f1"),
        "pair_precision": model.get("pair_precision"),
        "pair_recall": model.get("pair_recall"),
        "valid_structure_rate": model.get("valid_structure_rate"),
        "all_dot_ratio": model.get("all_dot_ratio"),
        "avg_pred_pair_count": avg_pred,
        "avg_true_pair_count": avg_true,
        "pair_count_ratio": avg_pred / avg_true if avg_true else None,
        "random_pair_f1": random_base.get("pair_f1"),
        "decode_method": benchmark.get("decode_method"),
        "last_gap": last_train.get("gap"),
        "last_rankAcc": last_train.get("rankAcc"),
        "last_loss": last_train.get("loss"),
        "analysis_keys": sorted(analysis.keys()),
        "diagnosis_keys": sorted(diagnosis.keys()),
        "train_log_rows": len(train_rows),
    }


def dataset_summary(path: Path, max_rows: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(path, limit=max_rows)
    lengths = [int(row.get("length") or len(row.get("seq", ""))) for row in rows]
    families = Counter(str(row.get("family") or "OTHER") for row in rows)
    all_dot = 0
    invalid = 0
    canonical = []
    pair_counts = []
    bad_examples = []
    for row in rows:
        seq = str(row.get("seq", "")).upper().replace("T", "U")
        struct = str(row.get("struct", ""))
        if set(struct) <= {"."}:
            all_dot += 1
        try:
            pairs = parse_dot_bracket(struct)
        except ValueError as exc:
            invalid += 1
            if len(bad_examples) < 10:
                bad_examples.append({"id": row.get("id"), "reason": str(exc)})
            pairs = []
        pair_counts.append(len(pairs))
        if pairs:
            ok = sum(1 for i, j in pairs if i < len(seq) and j < len(seq) and canonical_pair(seq[i], seq[j]))
            canonical.append(ok / len(pairs))
        if struct and len(seq) == len(struct) and not validate_structure(seq, struct):
            invalid += 1
    return {
        "input": str(path),
        "samples": len(rows),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": statistics.mean(lengths) if lengths else 0.0,
        "length_p50": statistics.median(lengths) if lengths else 0.0,
        "family_top20": families.most_common(20),
        "all_dot_ratio": all_dot / len(rows) if rows else 0.0,
        "invalid_count": invalid,
        "avg_pair_count": statistics.mean(pair_counts) if pair_counts else 0.0,
        "canonical_pair_ratio": statistics.mean(canonical) if canonical else 0.0,
        "bad_examples": bad_examples,
    }


class RNAAnalysisAgent:
    """LLM-backed analysis agent that never participates in model inference."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def call(self, user_prompt: str) -> str:
        if self.dry_run:
            return "# Dry Run Prompt\n\n" + user_prompt
        settings = self._settings()
        payload = {
            "model": settings["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
        }
        request = urllib.request.Request(
            settings["base_url"] + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + settings["api_key"],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"LLM request failed with HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"LLM request failed: {exc.reason}") from exc
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit("LLM response did not match OpenAI-compatible chat/completions format.") from exc

    def diagnose(self, run_dir: Path) -> tuple[dict[str, Any], str]:
        summary = compact_run(run_dir)
        prompt = f"""Role: Failure Diagnoser.

Diagnose this RNA-OmniDiffusion run.

Metrics JSON:
{json.dumps(summary, indent=2)}

Return:
1. overall health
2. likely failure modes
3. whether pair precision, recall, pair-count ratio, gap, and rankAcc are consistent
4. safe next experiment suggestions
5. what must not be claimed in a paper

Do not compare against external literature benchmarks unless supplied in the JSON.
Do not call the run unhealthy solely because it is below an external threshold.
Use only internal comparisons visible in the JSON, such as random baseline, valid rate, pair-count ratio, gap, and rankAcc.
"""
        return {"summary": summary, "dry_run": self.dry_run}, self.call(prompt)

    def schedule(self, run_dir: Path, config_path: Path | None = None) -> tuple[dict[str, Any], str]:
        summary = compact_run(run_dir)
        config_text = read_text(config_path, max_chars=12000) if config_path else ""
        prompt = f"""Role: Experiment Scheduler.

Create a conservative experiment schedule for RNA-OmniDiffusion.

Current run summary:
{json.dumps(summary, indent=2)}

Current config:
```yaml
{config_text}
```

Rules:
- Do not modify labels, predictions, or benchmark metrics.
- Do not introduce LLM inference, RNA-FM, RNA 3D, ligand, or protein tasks.
- Keep strict Nussinov as the final benchmark path.
- Use only existing CLI patterns:
  - python main.py train --config <config> --device cuda
  - python scripts/eval.py bench --config <config> --ckpt <ckpt> --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
  - python scripts/eval.py bench --config <config> --ckpt <ckpt> --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
  - python scripts/run.py external --configs <configs> --dataset <name> --split random --device cuda --decode nussinov --bench_workers 8 --tag <tag>
- Existing configs you may reference as executable commands: config/candidate.yaml, config/candidate_norefine.yaml, config/candidate_oldmask.yaml, config/oldbase.yaml, config/fixed.yaml, config/scan.json.
- If a new config is needed, write "manual config patch required" and give a YAML patch. Do not put nonexistent config paths such as config/exp1.yaml in runnable commands.
- Prefer 3 to 6 experiments with commands and stop criteria.

Return a schedule table and a fail-fast checklist.
"""
        return {"summary": summary, "config": str(config_path) if config_path else None, "dry_run": self.dry_run}, self.call(prompt)

    def report(self, inputs: list[Path], max_chars: int = 20000) -> tuple[dict[str, Any], str]:
        sources = {str(path): read_text(path, max_chars=max_chars) for path in inputs}
        prompt = f"""Role: Report Generator.

Draft a paper-ready report section for RNA-OmniDiffusion.

Use only the supplied source text. Do not invent new results.
Keep supported claims separate from unsupported claims.

Sources:
{json.dumps(sources, indent=2)}

Return:
1. concise model summary
2. main results paragraph
3. ablation paragraph
4. limitations paragraph
5. reviewer-risk checklist
"""
        return {"inputs": [str(path) for path in inputs], "dry_run": self.dry_run}, self.call(prompt)

    def auditdata(self, inputs: list[Path], max_rows: int | None = None) -> tuple[dict[str, Any], str]:
        summaries = [dataset_summary(path, max_rows=max_rows) for path in inputs]
        prompt = f"""Role: Data Quality Auditor.

Audit RNA JSONL dataset summaries for paper-quality risks.

Summaries:
{json.dumps(summaries, indent=2)}

Return:
1. data health assessment
2. leakage or split risks visible from the summary
3. structure-label risks
4. what extra checks should be run before paper submission
5. concise action list
"""
        return {"summaries": summaries, "dry_run": self.dry_run}, self.call(prompt)

    def _settings(self) -> dict[str, Any]:
        env = load_env()
        api_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
        base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = env.get("LLM_MODEL") or env.get("OPENAI_MODEL") or "gpt-4.1-mini"
        temperature = float(env.get("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(env.get("LLM_MAX_TOKENS", "1200"))
        if not api_key or api_key == "YOUR_KEY":
            raise SystemExit("Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY in .env.")
        if not model or model == "YOUR_MODEL_ID":
            raise SystemExit("Missing LLM model. Set LLM_MODEL in .env.")
        return {
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
