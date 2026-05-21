import argparse
from enum import Enum
from pathlib import Path

from .enums import (
    AssemblyType,
    ColorType,
    CompareWithOtherSegments,
    ScoreAlgorithm,
    ScoreMode,
)
from .paths import IMAGE_INPUT_DIR, IMAGE_OUTPUT_DIR


class SolverHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _expand_help(self, action):
        original_default = action.default
        if isinstance(original_default, Enum):
            action.default = original_default.name.lower()
        try:
            return super()._expand_help(action)
        finally:
            action.default = original_default


def enumNames(enum_type):
    return ", ".join(value.name.lower() for value in enum_type)


def enumValue(enum_type):
    def parse(value):
        normalized = value.strip().upper().replace("-", "_")
        try:
            return enum_type[normalized]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(
                f"choose one of: {enumNames(enum_type)}") from exc

    return parse


def buildArgumentParser():
    parser = argparse.ArgumentParser(
        description="Solve a square jigsaw puzzle from an input image.",
        formatter_class=SolverHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        default=IMAGE_INPUT_DIR / "William.png",
        help="Input image to split into square puzzle pieces.",
    )
    parser.add_argument(
        "-l",
        "--piece-size",
        "--length",
        type=int,
        default=30,
        help="Width and height, in pixels, of each square puzzle piece.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMAGE_OUTPUT_DIR,
        help="Directory for saved pieces and assembly images.",
    )
    parser.add_argument(
        "--output-name",
        default="test",
        help="Filename prefix for saved assembly-round images.",
    )
    parser.add_argument(
        "--save-segments",
        "--save-pieces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write each image segment to the output directory.",
    )
    parser.add_argument(
        "--save-assembly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an assembly image after each successful join.",
    )
    parser.add_argument(
        "--animation",
        "--show-animation",
        dest="show_animation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a Tkinter window that updates as the puzzle assembles.",
    )
    parser.add_argument(
        "--progress",
        "--print-progress",
        dest="show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print score calculation, best-buddy, and assembly progress.",
    )
    parser.add_argument(
        "--best-buddy",
        dest="connect_best_buddy_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the mutual best-buddy pre-assembly pass before Kruskal or Prim.",
    )
    parser.add_argument(
        "--kruskal-priority-queue",
        dest="use_kruskal_priority_queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the optimized priority-queue implementation for Kruskal assembly.",
    )
    parser.add_argument(
        "--boost-big-piece-priority",
        action="store_true",
        help="Favor joins with more touching edges when scoring component merges.",
    )
    parser.add_argument(
        "--trim-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim the greedy tree to the puzzle frame and fill remaining holes.",
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=None,
        help="Worker count for score calculation. Omit for automatic CPU-based selection.",
    )
    parser.add_argument(
        "--score-executor",
        choices=("serial", "thread", "process"),
        default="process",
        help="Executor backend for score calculation.",
    )
    parser.add_argument(
        "--color-type",
        type=enumValue(ColorType),
        metavar=f"{{{enumNames(ColorType)}}}",
        default=ColorType.LAB,
        help="Color space used before scoring.",
    )
    parser.add_argument(
        "--assembly-type",
        type=enumValue(AssemblyType),
        metavar=f"{{{enumNames(AssemblyType)}}}",
        default=AssemblyType.KRUSKAL,
        help="Assembly strategy.",
    )
    parser.add_argument(
        "--score-algorithm",
        type=enumValue(ScoreAlgorithm),
        metavar=f"{{{enumNames(ScoreAlgorithm)}}}",
        default=ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
        help="Underlying edge distance algorithm.",
    )
    parser.add_argument(
        "--score-mode",
        type=enumValue(ScoreMode),
        metavar=f"{{{enumNames(ScoreMode)}}}",
        default=ScoreMode.DISSIMILARITY,
        help="Use raw dissimilarity costs or second-best reliability costs.",
    )
    parser.add_argument(
        "--compare-type",
        type=enumValue(CompareWithOtherSegments),
        metavar=f"{{{enumNames(CompareWithOtherSegments)}}}",
        default=CompareWithOtherSegments.ONLY_BEST,
        help="Connection comparison formula used while choosing joins.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic segment shuffle order.",
    )
    return parser


def parseArguments(argv=None):
    args = buildArgumentParser().parse_args(argv)
    if args.piece_size <= 0:
        raise SystemExit("--piece-size must be greater than 0")
    if args.score_workers is not None and args.score_workers <= 0:
        raise SystemExit("--score-workers must be greater than 0")
    return args
