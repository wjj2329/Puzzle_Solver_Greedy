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
        "--gallagher-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the fixed-orientation Gallagher-style baseline: RGB, MGC, "
            "second-best reliability, Kruskal forest assembly, no best-buddy, "
            "and no trim/fill cleanup."
        ),
    )
    parser.add_argument(
        "--quality-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print paper-style assembly quality metrics after solving.",
    )
    parser.add_argument(
        "--rank-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include true-neighbor top-1/top-2/top-5 score ranks in the quality report.",
    )
    parser.add_argument(
        "--diagnostic-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print error diagnostics: largest correct islands, direct-placement "
            "mismatches, and low-score false seams."
        ),
    )
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        default=12,
        help="Maximum false seams shown by --diagnostic-report.",
    )
    parser.add_argument(
        "--gallagher-pairwise-kruskal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Gallagher-style pairwise edge-ordered Kruskal assembly "
            "instead of rescoring whole component pairs."
        ),
    )
    parser.add_argument(
        "--gallagher-edge-candidates",
        type=int,
        default=10,
        help="Top candidate pieces per piece side to queue for pairwise Gallagher Kruskal.",
    )
    parser.add_argument(
        "--gallagher-mutual-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Queue only reciprocal best edges in Gallagher pairwise Kruskal.",
    )
    parser.add_argument(
        "--growing-consensus-kruskal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Son-style 2x2 loop consensus edges for Kruskal assembly "
            "instead of raw pairwise Gallagher edges."
        ),
    )
    parser.add_argument(
        "--growing-consensus-edge-candidates",
        type=int,
        default=10,
        help="Top candidate pieces per piece side used to build consensus loops.",
    )
    parser.add_argument(
        "--growing-consensus-min-support",
        type=int,
        default=1,
        help="Minimum 2x2 consensus-loop support required for an assembly edge.",
    )
    parser.add_argument(
        "--growing-consensus-max-edges",
        type=int,
        default=25000,
        help=(
            "Maximum consensus-supported edges queued for assembly. Use 0 for "
            "no cap."
        ),
    )
    parser.add_argument(
        "--growing-consensus-priority",
        choices=("score", "support"),
        default="score",
        help=(
            "How consensus edges are ordered after meeting the support "
            "threshold."
        ),
    )
    parser.add_argument(
        "--growing-consensus-propose-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use three edges of an incomplete 2x2 loop to propose the missing "
            "fourth edge."
        ),
    )
    parser.add_argument(
        "--growing-consensus-fallback-pairwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After consensus-supported joins are exhausted, continue with "
            "ordinary pairwise Gallagher Kruskal edges."
        ),
    )
    parser.add_argument(
        "--repair-bad-joins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After pairwise Gallagher assembly, split internal seams above "
            "a score limit and re-merge the resulting components."
        ),
    )
    parser.add_argument(
        "--repair-bad-join-score-limit",
        type=float,
        default=1.0,
        help="Maximum internal seam score retained by --repair-bad-joins.",
    )
    parser.add_argument(
        "--repair-remerge-score-limit",
        type=float,
        default=None,
        help=(
            "Maximum pairwise edge score allowed when re-merging repaired "
            "components. Omit to reuse --repair-bad-join-score-limit."
        ),
    )
    parser.add_argument(
        "--repair-remerge-strategy",
        choices=("multi-contact", "pairwise"),
        default="multi-contact",
        help=(
            "How repaired fragments are re-merged. 'multi-contact' only "
            "accepts component joins with multiple touching seams; 'pairwise' "
            "reuses Gallagher edge-ordered one-seam remerge."
        ),
    )
    parser.add_argument(
        "--repair-iterations",
        type=int,
        default=1,
        help="Number of split/re-merge repair passes.",
    )
    parser.add_argument(
        "--repair-frame-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After repair remerge, place remaining fragments into the final "
            "frame using boundary-contact scoring."
        ),
    )
    parser.add_argument(
        "--repair-frame-placement-min-contacts",
        type=int,
        default=2,
        help="Minimum boundary contacts required for repair frame placement.",
    )
    parser.add_argument(
        "--repair-consensus-shifts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split low-consensus seams and try small whole-component shifts "
            "accepted by multi-contact boundary agreement."
        ),
    )
    parser.add_argument(
        "--repair-consensus-top-k",
        type=int,
        default=8,
        help="Top candidate rank used when deciding whether a seam has mutual support.",
    )
    parser.add_argument(
        "--repair-consensus-min-local-support",
        type=int,
        default=1,
        help="Minimum neighboring mutual-top-K 2x2 loop support that can preserve a seam.",
    )
    parser.add_argument(
        "--repair-consensus-min-contacts",
        type=int,
        default=6,
        help="Minimum boundary contacts required to accept a shifted component.",
    )
    parser.add_argument(
        "--repair-consensus-max-shift",
        type=int,
        default=1,
        help="Maximum row/column translation tested for each consensus component.",
    )
    parser.add_argument(
        "--repair-consensus-remerge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After consensus shift repair, re-merge remaining fragments using "
            "multi-contact Kruskal joins."
        ),
    )
    parser.add_argument(
        "--repair-consensus-frame-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After consensus remerge, strip empty borders and place remaining "
            "fragments into the final puzzle frame."
        ),
    )
    parser.add_argument(
        "--repair-consensus-frame-min-contacts",
        type=int,
        default=3,
        help="Minimum boundary contacts required for consensus frame placement.",
    )
    parser.add_argument(
        "--best-buddy",
        dest="connect_best_buddy_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the mutual best-buddy pre-assembly pass before Kruskal or Prim.",
    )
    parser.add_argument(
        "--prim-seed-strategy",
        choices=("neighborhood", "random"),
        default="neighborhood",
        help=(
            "How Prim chooses its initial root. 'neighborhood' prefers a "
            "piece/component with several strong outgoing edges; 'random' "
            "preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--prim-seed-neighbors",
        type=int,
        default=4,
        help="Outgoing edge count averaged by the Prim neighborhood seed.",
    )
    parser.add_argument(
        "--prim-priority-queue",
        dest="use_prim_priority_queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a frontier priority queue for Prim assembly.",
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
        "--staged-kruskal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefer component merges with at least two touching edges before "
            "falling back to ordinary one-edge Kruskal joins."
        ),
    )
    parser.add_argument(
        "--staged-kruskal-score-limit",
        type=float,
        default=None,
        help=(
            "Maximum score allowed in the multi-contact stage. Omit to use "
            "1.0 in reliability mode and no limit in dissimilarity mode."
        ),
    )
    parser.add_argument(
        "--relax-frame-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow Kruskal assembly to build an oversized collision-free tree "
            "before trimming to the final frame."
        ),
    )
    parser.add_argument(
        "--endgame-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Try relaxed boundary-to-boundary joins among the last few "
            "components before trim/fill."
        ),
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Keep this many alternate Kruskal assembly states. 1 uses greedy assembly.",
    )
    parser.add_argument(
        "--beam-candidates",
        type=int,
        default=None,
        help="Candidate joins to branch from each beam state. Omit to match beam width.",
    )
    parser.add_argument(
        "--beam-start-components",
        type=int,
        default=None,
        help=(
            "Run greedy Kruskal queue assembly until this many components "
            "remain, then switch to beam search. Omit for full beam search."
        ),
    )
    parser.add_argument(
        "--trim-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim the greedy tree to the puzzle frame and fill remaining holes.",
    )
    parser.add_argument(
        "--trim-fill-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try to place leftover assembled components as units before filling individual pieces.",
    )
    parser.add_argument(
        "--trim-fill-conservative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use assembly edge confidence to choose the trim frame and skip "
            "low-confidence hole fills."
        ),
    )
    parser.add_argument(
        "--trim-fill-border",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefer trim frames that place weakly matched piece edges on the "
            "outside border."
        ),
    )
    parser.add_argument(
        "--trim-fill-component-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Search final-frame placements of leftover assembled components "
            "before individual-piece filling."
        ),
    )
    parser.add_argument(
        "--trim-fill-edge-preserving",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Choose trim frames by retained assembled edges first and only "
            "fill holes with at least two occupied neighbors."
        ),
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
        "--score-storage",
        choices=("dense", "dict"),
        default="dense",
        help="Storage backend for pairwise edge scores.",
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
        "--symmetric-compatibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Average reciprocal directed edge scores so each seam uses both "
            "pieces' compatibility confidence."
        ),
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


def applyGallagherMode(args):
    if not args.gallagher_mode:
        return args
    args.color_type = ColorType.RGB
    args.score_algorithm = ScoreAlgorithm.MGC_DISTANCE
    args.score_mode = ScoreMode.RELIABILITY
    args.symmetric_compatibility = True
    args.gallagher_pairwise_kruskal = True
    args.assembly_type = AssemblyType.KRUSKAL
    args.compare_type = CompareWithOtherSegments.ONLY_BEST
    args.connect_best_buddy_first = False
    args.use_kruskal_priority_queue = True
    args.boost_big_piece_priority = False
    args.staged_kruskal = False
    args.relax_frame_bounds = False
    args.endgame_search = False
    args.beam_width = 1
    args.beam_candidates = None
    args.beam_start_components = None
    args.trim_fill = False
    args.trim_fill_components = False
    args.trim_fill_conservative = False
    args.trim_fill_border = False
    args.trim_fill_component_frame = False
    args.trim_fill_edge_preserving = False
    args.quality_report = True
    return args


def parseArguments(argv=None):
    args = applyGallagherMode(buildArgumentParser().parse_args(argv))
    if args.piece_size <= 0:
        raise SystemExit("--piece-size must be greater than 0")
    if args.score_workers is not None and args.score_workers <= 0:
        raise SystemExit("--score-workers must be greater than 0")
    if args.gallagher_edge_candidates <= 0:
        raise SystemExit("--gallagher-edge-candidates must be greater than 0")
    if args.growing_consensus_edge_candidates <= 0:
        raise SystemExit(
            "--growing-consensus-edge-candidates must be greater than 0")
    if args.growing_consensus_min_support <= 0:
        raise SystemExit(
            "--growing-consensus-min-support must be greater than 0")
    if args.growing_consensus_max_edges < 0:
        raise SystemExit("--growing-consensus-max-edges must be non-negative")
    if args.diagnostic_limit <= 0:
        raise SystemExit("--diagnostic-limit must be greater than 0")
    if args.repair_bad_join_score_limit < 0:
        raise SystemExit("--repair-bad-join-score-limit must be non-negative")
    if (
            args.repair_remerge_score_limit is not None
            and args.repair_remerge_score_limit < 0):
        raise SystemExit("--repair-remerge-score-limit must be non-negative")
    if args.repair_iterations <= 0:
        raise SystemExit("--repair-iterations must be greater than 0")
    if args.repair_frame_placement_min_contacts <= 0:
        raise SystemExit(
            "--repair-frame-placement-min-contacts must be greater than 0")
    if args.repair_consensus_top_k <= 0:
        raise SystemExit("--repair-consensus-top-k must be greater than 0")
    if args.repair_consensus_min_local_support < 0:
        raise SystemExit(
            "--repair-consensus-min-local-support must be non-negative")
    if args.repair_consensus_min_contacts <= 0:
        raise SystemExit("--repair-consensus-min-contacts must be greater than 0")
    if args.repair_consensus_max_shift < 0:
        raise SystemExit("--repair-consensus-max-shift must be non-negative")
    if args.repair_consensus_frame_min_contacts <= 0:
        raise SystemExit(
            "--repair-consensus-frame-min-contacts must be greater than 0")
    if (
            args.repair_bad_joins
            and (
                args.assembly_type != AssemblyType.KRUSKAL
                or not args.gallagher_pairwise_kruskal
            )):
        raise SystemExit(
            "--repair-bad-joins is only supported with Gallagher pairwise Kruskal")
    if (
            args.repair_consensus_shifts
            and (
                args.assembly_type != AssemblyType.KRUSKAL
                or not (
                    args.gallagher_pairwise_kruskal
                    or args.growing_consensus_kruskal
                )
            )):
        raise SystemExit(
            "--repair-consensus-shifts is only supported with "
            "Gallagher pairwise or growing-consensus Kruskal")
    if (
            args.growing_consensus_kruskal
            and args.assembly_type != AssemblyType.KRUSKAL):
        raise SystemExit(
            "--growing-consensus-kruskal is only supported with Kruskal assembly")
    if args.prim_seed_neighbors <= 0:
        raise SystemExit("--prim-seed-neighbors must be greater than 0")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be greater than 0")
    if args.beam_candidates is not None and args.beam_candidates <= 0:
        raise SystemExit("--beam-candidates must be greater than 0")
    if args.beam_start_components is not None and args.beam_start_components <= 1:
        raise SystemExit("--beam-start-components must be greater than 1")
    if args.beam_width > 1 and args.assembly_type != AssemblyType.KRUSKAL:
        raise SystemExit("--beam-width is only supported with Kruskal assembly")
    if args.staged_kruskal and args.assembly_type != AssemblyType.KRUSKAL:
        raise SystemExit("--staged-kruskal is only supported with Kruskal assembly")
    if args.staged_kruskal and args.beam_width > 1:
        raise SystemExit("--staged-kruskal cannot be combined with --beam-width greater than 1")
    if (
            args.staged_kruskal_score_limit is not None
            and args.staged_kruskal_score_limit < 0):
        raise SystemExit("--staged-kruskal-score-limit must be non-negative")
    if args.beam_start_components is not None and args.beam_width <= 1:
        raise SystemExit(
            "--beam-start-components requires --beam-width greater than 1")
    if (
            args.beam_start_components is not None
            and args.assembly_type != AssemblyType.KRUSKAL):
        raise SystemExit(
            "--beam-start-components is only supported with Kruskal assembly")
    return args
