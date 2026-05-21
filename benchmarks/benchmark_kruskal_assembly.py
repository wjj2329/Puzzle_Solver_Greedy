import argparse
import contextlib
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import puzzle_solver as Solver  # noqa: E402


def build_image(image_size, seed, image_mode):
    if image_mode == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    if image_mode == "gradient":
        y, x = np.indices((image_size, image_size))
        offset = seed % 256
        return np.stack(
            [
                (x + offset) % 256,
                (y + offset) % 256,
                ((x + y) // 2 + offset) % 256,
            ],
            axis=2,
        ).astype(np.uint8)
    raise ValueError(f"unknown image mode: {image_mode}")


def build_prepared_segments(
        image_size,
        piece_size,
        seed,
        image_mode,
        score_algorithm,
        score_mode,
        run_best_buddy):
    image = build_image(image_size, seed, image_mode)
    segments = Solver.breakUpImage(
        image,
        piece_size,
        save_segments=False,
        color_type=Solver.ColorType.RGB,
        score_algorithm=score_algorithm,
    )
    original_size = len(segments)
    with contextlib.redirect_stdout(io.StringIO()):
        Solver.calculateScores(
            segments,
            score_algorithm,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
    Solver.finalizeScores(segments, score_algorithm, score_mode)
    if run_best_buddy:
        with contextlib.redirect_stdout(io.StringIO()):
            Solver.connectBestBudsFirst(
                segments,
                original_size=original_size,
                show_progress=False,
            )
    return segments, original_size


def time_kruskal_assembly_run(
        image_size,
        piece_size,
        seed,
        image_mode,
        score_algorithm,
        score_mode,
        run_best_buddy,
        strategy,
        boost_priority_of_big_pieces_joining):
    segments, original_size = build_prepared_segments(
        image_size,
        piece_size,
        seed,
        image_mode,
        score_algorithm,
        score_mode,
        run_best_buddy,
    )
    starting_components = len(segments)
    rounds = 0
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if strategy == "queue":
            rounds = Solver.assembleKruskalWithPriorityQueue(
                segments,
                original_size,
                boost_priority_of_big_pieces_joining,
            )
        else:
            while len(segments) > 1:
                best_connection = Solver.findBestConnectionKruskal(
                    segments,
                    Solver.CompareWithOtherSegments.ONLY_BEST,
                    boost_priority_of_big_pieces_joining,
                    Solver.CompareWithOtherSegments.ONLY_BEST,
                )
                if best_connection.pic_connection_matrix is None:
                    break
                Solver.joinPieces(best_connection, segments, original_size)
                rounds += 1
    elapsed = time.perf_counter() - started
    return elapsed, original_size, starting_components, rounds, len(segments)


def parse_score_algorithm(name):
    normalized = name.strip().upper().replace("-", "_")
    try:
        return Solver.ScoreAlgorithm[normalized]
    except KeyError as exc:
        choices = ", ".join(algorithm.name.lower()
                            for algorithm in Solver.ScoreAlgorithm)
        raise argparse.ArgumentTypeError(
            f"unknown score algorithm {name!r}; choose one of: {choices}") from exc


def parse_score_mode(name):
    normalized = name.strip().upper().replace("-", "_")
    try:
        return Solver.ScoreMode[normalized]
    except KeyError as exc:
        choices = ", ".join(mode.name.lower() for mode in Solver.ScoreMode)
        raise argparse.ArgumentTypeError(
            f"unknown score mode {name!r}; choose one of: {choices}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the round-by-round Kruskal assembly loop.")
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--piece-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--strategy",
        choices=("scan", "queue", "both"),
        default="both",
        help="Assembly strategy to time. scan is the old full-scan loop; "
             "queue is the priority-queue implementation.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("gradient", "random"),
        default="gradient",
        help="Synthetic image source. Gradient is stable for full assembly; "
             "random can expose solver failure cases.",
    )
    parser.add_argument(
        "--score-algorithm",
        type=parse_score_algorithm,
        default=Solver.ScoreAlgorithm.EUCLIDEAN,
        help="Score algorithm to use for precomputed edge scores. "
             "Default: euclidean",
    )
    parser.add_argument(
        "--score-mode",
        type=parse_score_mode,
        default=Solver.ScoreMode.DISSIMILARITY,
        help="Score interpretation mode. dissimilarity preserves raw lower-is-better "
             "edge costs; reliability uses second-best reliability costs.",
    )
    parser.add_argument(
        "--skip-best-buddy",
        action="store_true",
        help="Time Kruskal assembly without running the best-buddy setup pass.",
    )
    parser.add_argument(
        "--boost-big-piece-priority",
        action="store_true",
        help="Enable the existing big-piece priority score adjustment.",
    )
    args = parser.parse_args()

    if args.image_size % args.piece_size != 0:
        raise SystemExit("--image-size must be divisible by --piece-size")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    segment_count = (args.image_size // args.piece_size) ** 2
    run_best_buddy = not args.skip_best_buddy
    strategies = ("scan", "queue") if args.strategy == "both" else (args.strategy,)
    print(
        f"segments: {segment_count} "
        f"({args.image_size}x{args.image_size}, piece={args.piece_size})"
    )
    print(f"image_mode: {args.image_mode}")
    print(f"score_algorithm: {args.score_algorithm.name.lower()}")
    print(f"score_mode: {args.score_mode.name.lower()}")
    print(f"best_buddy_setup: {run_best_buddy}")
    print("setup: score calculation and best-buddy are performed before timing")
    for strategy in strategies:
        timings = []
        component_counts = []
        round_counts = []
        remaining_counts = []
        print(f"strategy: {strategy}")
        for run_index in range(args.repeat):
            elapsed, original_size, starting_components, rounds, remaining = (
                time_kruskal_assembly_run(
                    args.image_size,
                    args.piece_size,
                    args.seed + run_index,
                    args.image_mode,
                    args.score_algorithm,
                    args.score_mode,
                    run_best_buddy,
                    strategy,
                    args.boost_big_piece_priority,
                )
            )
            timings.append(elapsed)
            component_counts.append(starting_components)
            round_counts.append(rounds)
            remaining_counts.append(remaining)
            print(
                f"  run {run_index + 1:02d}: "
                f"{elapsed:.6f}s "
                f"components={starting_components}/{original_size} "
                f"rounds={rounds} "
                f"remaining={remaining}"
            )
        print(
            f"  best={min(timings):.6f}s "
            f"median={statistics.median(timings):.6f}s "
            f"mean={statistics.mean(timings):.6f}s "
            f"components_median={statistics.median(component_counts)} "
            f"rounds_median={statistics.median(round_counts)} "
            f"remaining_median={statistics.median(remaining_counts)}"
        )


if __name__ == "__main__":
    main()
