"""Direct-file entry point for the puzzle_solver package.

Prefer running ``python3 -m puzzle_solver`` from the repository root. This file
exists for people who want an explicit script path.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzle_solver import *  # noqa: F401,F403,E402
from puzzle_solver.runner import main  # noqa: E402


if __name__ == "__main__":
    main()
