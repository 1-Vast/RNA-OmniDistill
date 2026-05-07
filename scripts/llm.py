from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.agent.analyzer import RNAAnalysisAgent, write_report


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Input directory not found: {path}")
    return path


def run_diagnose(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.diagnose(require_dir(Path(args.run)))
    write_report(Path(args.out), "diagnose", data, markdown)


def run_schedule(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    config = require_file(Path(args.config)) if args.config else None
    data, markdown = agent.schedule(require_dir(Path(args.run)), config)
    write_report(Path(args.out), "schedule", data, markdown)


def run_report(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.report([require_file(Path(path)) for path in args.inputs], max_chars=args.max_chars)
    write_report(Path(args.out), "report", data, markdown)


def run_auditdata(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.auditdata([require_file(Path(path)) for path in args.inputs], max_rows=args.max_rows)
    write_report(Path(args.out), "dataaudit", data, markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM analysis agent for RNA-OmniDiffusion. "
            "It only reads artifacts and writes reports; it never runs benchmark inference."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    common.add_argument("--dry_run", action="store_true")

    diagnose = sub.add_parser("diagnose", parents=[common], help="Diagnose one run directory.")
    diagnose.add_argument("--run", required=True)
    diagnose.set_defaults(func=run_diagnose)

    schedule = sub.add_parser("schedule", parents=[common], help="Suggest safe next experiments.")
    schedule.add_argument("--run", required=True)
    schedule.add_argument("--config")
    schedule.set_defaults(func=run_schedule)

    report = sub.add_parser("report", parents=[common], help="Generate a paper report draft from existing files.")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--max_chars", type=int, default=20000)
    report.set_defaults(func=run_report)

    auditdata = sub.add_parser("auditdata", parents=[common], help="Audit RNA JSONL dataset summaries.")
    auditdata.add_argument("--inputs", nargs="+", required=True)
    auditdata.add_argument("--max_rows", type=int)
    auditdata.set_defaults(func=run_auditdata)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
