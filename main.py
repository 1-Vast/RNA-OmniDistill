"""RNA-OmniDiffusion command line interface.

Provides training, evaluation, inference, smoke tests,
parameter inspection, framework overview, and Agent help.

Quick reference::

    python main.py overview          Show framework overview
    python main.py train --config config/candidate.yaml --device cuda
    python main.py eval --config config/candidate.yaml --ckpt outputs/candidate/best.pt
    python main.py infer --config config/candidate.yaml --ckpt outputs/candidate/best.pt --task seq2struct --seq GCAUAGC
    python main.py smoke             Run tiny CPU sanity test
    python main.py params --config config/candidate.yaml
    python main.py agent             Show Agent shell usage

See docs/usage.md for full documentation.
"""

from __future__ import annotations

from models.cli import build_parser

__version__ = "1.0.0"


def main(argv=None):
    """Parse command-line arguments and dispatch to the appropriate handler.

    The parser and all subcommand handlers live in :mod:`models.cli`.
    torch-dependent commands (train, eval, infer, smoke) use lazy imports
    so that lightweight commands (overview, models, params, agent) work
    without a GPU or even without PyTorch installed.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  ``None`` means ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
