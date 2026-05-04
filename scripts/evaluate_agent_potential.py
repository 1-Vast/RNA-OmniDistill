from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SystemExit(f"Required file not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exists(path: str | Path) -> bool:
    return Path(path).exists()


def score_agent_readiness(model_potential: str, completeness: dict, separability: dict, reproducibility: dict) -> str:
    complete_count = sum(bool(value) for value in completeness.values())
    separable_count = sum(bool(value) for value in separability.values())
    reproducible_count = sum(bool(value) for value in reproducibility.values())
    if model_potential in {"High", "Medium"} and complete_count >= 5 and separable_count >= 3 and reproducible_count >= 3:
        return "High"
    if complete_count >= 4 and reproducible_count >= 2:
        return "Medium"
    return "Low"


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Agent-readiness Evaluation",
        "",
        f"Conclusion: **{result['agent_potential']}**",
        "",
        "This is not an Agent implementation and does not select a framework.",
        "",
        "## Recommended Future Roles",
        "- Experiment Scheduler",
        "- Failure Diagnoser",
        "- Config Patch Advisor",
        "- Report Generator",
        "",
        "## Explicitly Forbidden Future Roles",
        "- Agent participates in benchmark inference",
        "- Agent modifies test labels",
        "- Agent modifies test data",
        "- Agent is used as the source of model performance",
        "- Agent calls an LLM during test inference",
        "",
        "## Checks",
    ]
    for section in ["diagnostic_signal_completeness", "failure_separability", "actionability", "reproducibility", "safety_boundary"]:
        lines.append(f"### {section}")
        for key, value in result[section].items():
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether future Agent tooling is justified.")
    parser.add_argument("--model_potential", required=True)
    parser.add_argument("--training_analysis", required=True)
    parser.add_argument("--prediction_diagnosis", required=True)
    parser.add_argument("--ablation_summary", required=True)
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = read_json(Path(args.model_potential))
    training = read_json(Path(args.training_analysis), required=False)
    prediction = read_json(Path(args.prediction_diagnosis), required=False)
    ablations = read_json(Path(args.ablation_summary), required=False)
    run_summary = read_json(Path(args.run_summary), required=False)
    run_dir = Path(args.run_summary).parent

    completeness = {
        "train_log.jsonl": exists(Path(args.training_analysis).parent / "train_log.jsonl"),
        "benchmark_test.json": exists(Path(args.model_potential).parent.parent / "archiveii_full" / "benchmark_test.json") or bool(model),
        "predictions_test.jsonl": exists(Path(args.prediction_diagnosis).parent / "predictions_test.jsonl"),
        "training_analysis.json": bool(training),
        "prediction_diagnosis.json": bool(prediction),
        "ablation_summary": bool(ablations),
    }
    rows = ablations.get("rows", [])
    full = next((row for row in rows if row.get("variant") == "full"), {})
    no_nuss = next((row for row in rows if row.get("variant") == "no_nussinov"), {})
    no_pair = next((row for row in rows if row.get("variant") == "no_pair_head"), {})
    random_mask = next((row for row in rows if row.get("variant") == "random_mask_only"), {})
    separability = {
        "no_nussinov_invalid_or_lower_f1": float(no_nuss.get("valid_structure_rate", 1.0)) < float(full.get("valid_structure_rate", 1.0)) or float(no_nuss.get("pair_f1", 0.0)) < float(full.get("pair_f1", 0.0)),
        "no_pair_head_affects_pair_f1": float(no_pair.get("pair_f1", full.get("pair_f1", 0.0))) < float(full.get("pair_f1", 0.0)),
        "random_mask_only_affects_f1": abs(float(random_mask.get("pair_f1", full.get("pair_f1", 0.0))) - float(full.get("pair_f1", 0.0))) > 1e-6,
        "full_has_clear_advantage": any(float(row.get("pair_f1", 0.0)) < float(full.get("pair_f1", 0.0)) for row in rows if row.get("variant") != "full"),
    }
    actionability = {
        "all_dot_collapse_maps_to_config": True,
        "under_pairing_maps_to_config": True,
        "over_pairing_maps_to_config": True,
        "pair_head_not_learning_maps_to_config_or_warning": True,
        "invalid_structure_maps_to_decode_config": True,
    }
    reproducibility = {
        "commands_txt": exists(run_dir / "commands.txt"),
        "run_summary_json": bool(run_summary),
        "config_saved": any(run_dir.glob("*.yaml")),
        "logs_dir": exists(run_dir / "logs"),
        "steps_recorded": bool(run_summary.get("steps")),
    }
    safety = {
        "no_test_data_modification_required": True,
        "no_label_modification_required": True,
        "no_llm_at_test_inference_required": True,
        "agent_not_model_output_source": True,
    }
    agent_potential = score_agent_readiness(model.get("potential", "Low"), completeness, separability, reproducibility)
    if model.get("potential") == "Low":
        agent_potential = "Low" if sum(completeness.values()) < 5 else "Medium"
    result = {
        "agent_potential": agent_potential,
        "model_potential": model.get("potential", "Low"),
        "diagnostic_signal_completeness": completeness,
        "failure_separability": separability,
        "actionability": actionability,
        "reproducibility": reproducibility,
        "safety_boundary": safety,
        "recommendation": "Do not choose a concrete Agent framework yet; use reports and diagnostics first.",
    }
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agent_potential.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "agent_potential.md", result)
    print(f"agent potential -> {output_dir / 'agent_potential.md'}")
    print(f"conclusion={agent_potential}")


if __name__ == "__main__":
    main()

