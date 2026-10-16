"""Entry point: ``python -m aiforge`` or the ``aiforge`` command."""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="aiforge", description="No-code neural network studio in your terminal.")
    parser.add_argument("-w", "--workspace", default=os.environ.get("AIFORGE_WORKSPACE", "workspace"),
                        help="Folder for models, datasets and exports (default: ./workspace)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # keep library warnings from drawing over the UI

    from .tui.app import AIForgeApp

    AIForgeApp(Path(args.workspace).expanduser().resolve()).run()


if __name__ == "__main__":
    main()
