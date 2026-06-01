import argparse
import contextlib
import io
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import imageio.v3 as iio
from skimage import color

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import puzzle_solver as Solver  # noqa: E402


class PhaseTimer:
    def __init__(self):
        self.timings = OrderedDict()

    @contextlib.contextmanager
    def time(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = (
                self.timings.get(name, 0.0) + time.perf_counter() - started
            )


def parse_enum(enum_type, name):
    normalized = name.strip().upper().replace("-", "_")
    try:
        return enum_type[normalized]
    except KeyError as exc:
        choices = ", ".join(item.name.lower() for item in enum_type)
        raise argparse.ArgumentTypeError(
            f"unknown value {name!r}; choose one of: {choices}") from exc


def quiet_stdout(show_progress):
    if show_progress:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def format_workers(workers):
    return "auto" if workers is None else str(workers)


def print_timings(timings, total_elapsed):
    print("phase timings:")
    for name, elapsed in sorted(
            timings.items(), key=lambda item: item[1], reverse=True):
        percent = (elapsed / total_elapsed) * 100 if total_elapsed else 0.0
        print(f"  {name:24s} {elapsed:9.3f}s {percent:6.1f}%")


def piece_position(piece_number, width):
    piece_index = piece_number - 1
    return divmod(piece_index, width)


def is_correct_neighbor(piece, neighbor, row_delta, col_delta):
    piece_row, piece_col = piece_position(
        piece.piece_number,
        piece.max_height,
    )
    neighbor_row, neighbor_col = piece_position(
        neighbor.piece_number,
        neighbor.max_height,
    )
    return (
        neighbor_row - piece_row == row_delta
        and neighbor_col - piece_col == col_delta
    )


def assembly_quality(segment_list):
    if not segment_list:
        return {
            "adjacent": 0,
            "correct": 0,
            "incorrect": 0,
            "possible": 0,
            "coverage": 0.0,
            "precision": 0.0,
        }

    frame_height = segment_list[0].max_height
    frame_width = segment_list[0].max_width
    possible = (
        frame_height * (frame_width - 1)
        + frame_width * (frame_height - 1)
    )
    adjacent = 0
    correct = 0
    for segment in segment_list:
        matrix = segment.pic_connection_matrix
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                piece = matrix[row, col]
                if piece == 0:
                    continue
                if col + 1 < cols and matrix[row, col + 1] != 0:
                    adjacent += 1
                    if is_correct_neighbor(piece, matrix[row, col + 1], 0, 1):
                        correct += 1
                if row + 1 < rows and matrix[row + 1, col] != 0:
                    adjacent += 1
                    if is_correct_neighbor(piece, matrix[row + 1, col], 1, 0):
                        correct += 1
    incorrect = adjacent - correct
    return {
        "adjacent": adjacent,
        "correct": correct,
        "incorrect": incorrect,
        "possible": possible,
        "coverage": correct / possible if possible else 0.0,
        "precision": correct / adjacent if adjacent else 0.0,
    }


def print_quality(label, quality):
    print(
        f"{label}: adjacent={quality['adjacent']} "
        f"correct={quality['correct']} incorrect={quality['incorrect']} "
        f"possible={quality['possible']} "
        f"coverage={quality['coverage']:.3f} "
        f"precision={quality['precision']:.3f}"
    )


def run_profile(args):
    timer = PhaseTimer()
    output_dir = args.output_dir
    image_path = args.image
    score_workers = args.score_workers
    color_type = parse_enum(Solver.ColorType, args.color_type)
    score_algorithm = parse_enum(Solver.ScoreAlgorithm, args.score_algorithm)
    score_mode = parse_enum(Solver.ScoreMode, args.score_mode)
    compare_type = parse_enum(
        Solver.CompareWithOtherSegments,
        args.compare_type,
    )
    total_started = time.perf_counter()

    with timer.time("load_image"):
        image = iio.imread(image_path)
    image_shape = image.shape

    if color_type == Solver.ColorType.LAB:
        with timer.time("convert_to_lab"):
            image = color.rgb2lab(image)

    with timer.time("break_up_image"):
        segments = Solver.breakUpImage(
            image,
            args.piece_size,
            save_segments=False,
            color_type=color_type,
            output_dir=output_dir,
            score_storage=args.score_storage,
        )
    if args.relax_frame_bounds:
        for segment in segments:
            segment.enforce_frame_bounds = False
    original_size = len(segments)
    segment_save_batch = None
    if args.save_segments:
        with timer.time("start_segment_writes"):
            segment_save_batch = Solver.saveSegmentImagesAsync(
                segments,
                color_type,
                output_dir,
            )

    with timer.time("calculate_scores"):
        with quiet_stdout(args.show_progress):
            Solver.calculateScores(
                segments,
                score_algorithm,
                show_progress=args.show_progress,
                max_workers=score_workers,
                executor_type=args.score_executor,
            )

    with timer.time("finalize_scores"):
        Solver.finalizeScores(segments, score_algorithm, score_mode)

    if segment_save_batch is not None:
        with timer.time("wait_segment_writes"):
            segment_save_batch.wait()

    with timer.time("shuffle_segments"):
        random.Random(args.seed).shuffle(segments)

    after_best_buddy = len(segments)
    if not args.skip_best_buddy:
        with timer.time("best_buddy"):
            with quiet_stdout(args.show_progress):
                Solver.connectBestBudsFirst(
                    segments,
                    original_size=original_size,
                    show_progress=args.show_progress,
                )
        after_best_buddy = len(segments)

    kruskal_queue = None
    if args.assembly_strategy == "queue":
        with timer.time("build_kruskal_queue"):
            kruskal_queue = Solver.KruskalConnectionPriorityQueue(
                segments,
                args.boost_big_piece_priority,
                compare_type,
                compare_type,
            )

    rounds = 0
    assembly_started = time.perf_counter()
    if args.assembly_strategy == "beam":
        def save_beam_join(best_connection, round_number):
            if not args.save_assembly:
                return
            Solver.saveImage(
                best_connection,
                args.piece_size,
                round_number,
                color_type,
                args.output_name,
                output_dir=output_dir,
            )

        if args.beam_start_components is None:
            with timer.time("assembly_beam_search"):
                rounds = Solver.assembleKruskalBeamSearch(
                    segments,
                    original_size,
                    beam_width=args.beam_width,
                    beam_candidates=args.beam_candidates,
                    boost_priority_of_big_pieces_joining=args.boost_big_piece_priority,
                    compare_type=compare_type,
                    compare_mode=compare_type,
                    show_progress=args.show_progress,
                    on_join=save_beam_join if args.save_assembly else None,
                )
        else:
            with timer.time("assembly_hybrid_beam_search"):
                rounds = Solver.assembleKruskalHybridBeamSearch(
                    segments,
                    original_size,
                    beam_start_components=args.beam_start_components,
                    beam_width=args.beam_width,
                    beam_candidates=args.beam_candidates,
                    boost_priority_of_big_pieces_joining=args.boost_big_piece_priority,
                    compare_type=compare_type,
                    compare_mode=compare_type,
                    show_progress=args.show_progress,
                    on_join=save_beam_join if args.save_assembly else None,
                )
    while len(segments) > 1 and args.assembly_strategy != "beam":
        if kruskal_queue is None:
            with timer.time("assembly_find_best"):
                best_connection = Solver.findBestConnectionKruskal(
                    segments,
                    compare_type,
                    args.boost_big_piece_priority,
                    compare_type,
                )
        else:
            with timer.time("assembly_pop_best"):
                best_connection = kruskal_queue.popBestConnection(segments)

        if best_connection.pic_connection_matrix is None:
            break

        with timer.time("assembly_join"):
            Solver.joinPieces(best_connection, segments, original_size)

        if kruskal_queue is not None:
            with timer.time("assembly_queue_update"):
                kruskal_queue.addConnectionsFor(
                    best_connection.own_segment,
                    segments,
                )

        if args.save_assembly:
            with timer.time("save_assembly_images"):
                Solver.saveImage(
                    best_connection,
                    args.piece_size,
                    rounds,
                    color_type,
                    args.output_name,
                    output_dir=output_dir,
                )
        rounds += 1

    assembly_elapsed = time.perf_counter() - assembly_started
    assembly_recorded = sum(
        elapsed
        for name, elapsed in timer.timings.items()
        if name.startswith("assembly_") or name == "save_assembly_images"
    )
    assembly_overhead = assembly_elapsed - assembly_recorded
    if assembly_overhead > 0.001:
        timer.timings["assembly_overhead"] = assembly_overhead

    assembly_remaining = len(segments)
    endgame_merges = 0
    if args.endgame_search:
        with timer.time("endgame_search"):
            with quiet_stdout(args.show_progress):
                endgame_merges = Solver.connectEndgameComponents(
                    segments,
                    show_progress=args.show_progress,
                )
    if args.trim_fill:
        with timer.time("trim_fill"):
            Solver.trimAndFillAssembly(
                segments,
                show_progress=args.show_progress,
                preserve_components=args.trim_fill_components,
                conservative_fill=args.trim_fill_conservative,
                border_tiebreak=args.trim_fill_border,
                component_frame_search=args.trim_fill_component_frame,
                edge_preserving=args.trim_fill_edge_preserving,
            )

    total_elapsed = time.perf_counter() - total_started
    print(f"image: {image_path}")
    print(f"image_shape: {image_shape}")
    print(f"piece_size: {args.piece_size}")
    print(f"pieces: {original_size}")
    print(f"after_best_buddy: {after_best_buddy}")
    print(f"assembly_remaining: {assembly_remaining}")
    print(f"endgame_merges: {endgame_merges}")
    print(f"remaining: {len(segments)}")
    print(f"rounds: {rounds}")
    print(f"color_type: {color_type.name.lower()}")
    print(f"score_algorithm: {score_algorithm.name.lower()}")
    print(f"score_mode: {score_mode.name.lower()}")
    print(f"compare_type: {compare_type.name.lower()}")
    print(f"score_storage: {args.score_storage}")
    print(
        f"score_executor: {args.score_executor} "
        f"workers={format_workers(score_workers)}"
    )
    print(f"assembly_strategy: {args.assembly_strategy}")
    print(f"relax_frame_bounds: {args.relax_frame_bounds}")
    print(f"endgame_search: {args.endgame_search}")
    if args.assembly_strategy == "beam":
        print(f"beam_width: {args.beam_width}")
        print(f"beam_candidates: {args.beam_candidates or args.beam_width}")
        print(f"beam_start_components: {args.beam_start_components}")
    if args.quality_report:
        print_quality("quality", assembly_quality(segments))
    print(f"save_segments: {args.save_segments}")
    print(f"save_assembly: {args.save_assembly}")
    print(f"trim_fill_components: {args.trim_fill_components}")
    print(f"trim_fill_conservative: {args.trim_fill_conservative}")
    print(f"trim_fill_border: {args.trim_fill_border}")
    print(f"trim_fill_component_frame: {args.trim_fill_component_frame}")
    print(f"trim_fill_edge_preserving: {args.trim_fill_edge_preserving}")
    print(f"total: {total_elapsed:.3f}s")
    print_timings(timer.timings, total_elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Profile a full puzzle solver run by major phase.")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "input_image" / "William.png",
        help="Input image to solve. Default: input_image/William.png",
    )
    parser.add_argument(
        "--piece-size",
        type=int,
        default=30,
        help="Square tile size. Default matches the solver runner.",
    )
    parser.add_argument(
        "--color-type",
        default="lab",
        help="Color type: rgb or lab. Default matches the solver runner.",
    )
    parser.add_argument(
        "--score-algorithm",
        default="euclidean_and_mahalanobis",
        help="Score algorithm name. Default matches the solver runner.",
    )
    parser.add_argument(
        "--score-mode",
        default="dissimilarity",
        help="Score interpretation mode: dissimilarity or reliability.",
    )
    parser.add_argument(
        "--compare-type",
        default="only_best",
        help="Join comparison formula: only_best or compare_with_second.",
    )
    parser.add_argument(
        "--score-executor",
        choices=("serial", "thread", "process"),
        default="process",
        help="Score calculation backend. Default matches the solver runner.",
    )
    parser.add_argument(
        "--score-storage",
        choices=("dense", "dict"),
        default="dense",
        help="Score storage backend. Default matches the solver runner.",
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=None,
        help="Score worker count. Omit for the solver's automatic choice.",
    )
    parser.add_argument(
        "--assembly-strategy",
        choices=("queue", "scan", "beam"),
        default="queue",
        help="Kruskal assembly strategy. Default uses the optimized queue.",
    )
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--beam-candidates", type=int, default=None)
    parser.add_argument("--beam-start-components", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--skip-best-buddy",
        action="store_true",
        help="Skip the best-buddy pre-assembly pass.",
    )
    parser.add_argument(
        "--boost-big-piece-priority",
        action="store_true",
        help="Enable the existing big-piece priority score adjustment.",
    )
    parser.add_argument(
        "--relax-frame-bounds",
        action="store_true",
        help=(
            "Allow Kruskal assembly to build an oversized collision-free tree "
            "before trim/fill."
        ),
    )
    parser.add_argument(
        "--endgame-search",
        action="store_true",
        help=(
            "Try relaxed boundary-to-boundary joins among the last few "
            "components before trim/fill."
        ),
    )
    parser.add_argument(
        "--save-segments",
        action="store_true",
        help="Write individual tile images while profiling.",
    )
    parser.add_argument(
        "--save-assembly",
        action="store_true",
        help="Write each assembly round image while profiling.",
    )
    parser.add_argument(
        "--trim-fill",
        action="store_true",
        help="Run and time trim/fill after assembly.",
    )
    parser.add_argument(
        "--trim-fill-components",
        action="store_true",
        help="Place leftover components as units before individual hole filling.",
    )
    parser.add_argument(
        "--trim-fill-conservative",
        action="store_true",
        help=(
            "Use assembly edge confidence to choose the trim frame and skip "
            "low-confidence hole fills."
        ),
    )
    parser.add_argument(
        "--trim-fill-border",
        action="store_true",
        help=(
            "Prefer trim frames that place weakly matched piece edges on the "
            "outside border."
        ),
    )
    parser.add_argument(
        "--trim-fill-component-frame",
        action="store_true",
        help=(
            "Search final-frame placements of leftover assembled components "
            "before individual-piece filling."
        ),
    )
    parser.add_argument(
        "--trim-fill-edge-preserving",
        action="store_true",
        help=(
            "Choose trim frames by retained assembled edges first and only "
            "fill holes with at least two occupied neighbors."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output_image" / "profile_run",
        help="Directory for optional segment and assembly images.",
    )
    parser.add_argument(
        "--output-name",
        default="profile",
        help="Filename prefix for optional assembly images.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show solver progress output while profiling.",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Report how many final neighboring piece pairs match the source image.",
    )
    args = parser.parse_args()

    if args.piece_size <= 0:
        raise SystemExit("--piece-size must be greater than 0")
    if args.score_workers is not None and args.score_workers <= 0:
        raise SystemExit("--score-workers must be greater than 0")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be greater than 0")
    if args.beam_candidates is not None and args.beam_candidates <= 0:
        raise SystemExit("--beam-candidates must be greater than 0")
    if args.beam_start_components is not None and args.beam_start_components <= 1:
        raise SystemExit("--beam-start-components must be greater than 1")
    if args.beam_start_components is not None and args.assembly_strategy != "beam":
        raise SystemExit(
            "--beam-start-components is only supported with --assembly-strategy beam")

    run_profile(args)


if __name__ == "__main__":
    main()
