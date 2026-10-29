"""Entry point: ``python -m aiforge`` or the ``aiforge`` command."""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from . import __version__


def default_workspace() -> Path:
    """Where models live unless -w / AIFORGE_WORKSPACE says otherwise.

    A ``./workspace`` folder in the current directory wins (that's where earlier versions and
    source checkouts keep their models); otherwise models go to ``~/AIForge``.
    """
    env = os.environ.get("AIFORGE_WORKSPACE")
    if env:
        return Path(env)
    local = Path.cwd() / "workspace"
    if (local / "models").is_dir():
        return local
    return Path.home() / "AIForge"


def build_parser() -> argparse.ArgumentParser:
    from .cli import add_commands

    parser = argparse.ArgumentParser(prog="aiforge", description="No-code neural network studio in your terminal.")
    parser.add_argument("-w", "--workspace", default=None,
                        help="Folder for models, datasets and exports "
                             "(default: ./workspace if it exists, otherwise ~/AIForge)")
    parser.add_argument("-V", "--version", action="version", version=f"aiforge {__version__}")
    # Lets -w also come after the command (aiforge train my-model -w ~/runs).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-w", "--workspace", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    add_commands(parser, common)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    warnings.filterwarnings("ignore")  # keep library warnings from drawing over the UI
    workspace = (Path(args.workspace) if args.workspace else default_workspace()).expanduser().resolve()

    if args.command:
        from .cli import run

        raise SystemExit(run(args, workspace))

    from .tui.app import AIForgeApp

    AIForgeApp(workspace).run()


if __name__ == "__main__":
    main()
