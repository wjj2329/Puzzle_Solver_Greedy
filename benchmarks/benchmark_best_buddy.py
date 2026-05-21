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


def build_scored_segments(image_size, piece_size, seed, score_algorithm, score_mode):
    rng = np.random.default_rng(seed)
    image = rng.integers(
        0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    segments = Solver.breakUpImage(
        image,
        piece_size,
        save_segments=False,
        color_type=Solver.ColorType.RGB,
        score_algorithm=score_algorithm,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        Solver.calculateScores(
            segments,
            score_algorithm,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
    Solver.finalizeScores(segments, score_algorithm, score_mode)
    return segments


def time_best_buddy_run(image_size, piece_size, seed, score_algorithm, score_mode):
    segments = build_scored_segments(
        image_size,
        piece_size,
        seed,
        score_algorithm,
        score_mode,
    )
    original_size = len(segments)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        Solver.connectBestBudsFirst(
            segments,
            original_size=original_size,
            show_progress=False,
        )
    elapsed = time.perf_counter() - started
    return elapsed, original_size, original_size - len(segments)


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
        description="Benchmark the best-buddy pre-assembly pass.")
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--piece-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=3)
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
    args = parser.parse_args()

    if args.image_size % args.piece_size != 0:
        raise SystemExit("--image-size must be divisible by --piece-size")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    segment_count = (args.image_size // args.piece_size) ** 2
    print(
        f"segments: {segment_count} "
        f"({args.image_size}x{args.image_size}, piece={args.piece_size})"
    )
    print(f"score_algorithm: {args.score_algorithm.name.lower()}")
    print(f"score_mode: {args.score_mode.name.lower()}")
    print("setup: score calculation is performed before timing each run")

    timings = []
    merge_counts = []
    for run_index in range(args.repeat):
        elapsed, original_size, merges = time_best_buddy_run(
            args.image_size,
            args.piece_size,
            args.seed + run_index,
            args.score_algorithm,
            args.score_mode,
        )
        timings.append(elapsed)
        merge_counts.append(merges)
        print(
            f"run {run_index + 1:02d}: "
            f"{elapsed:.6f}s merges={merges}/{original_size}"
        )

    print(
        f"best={min(timings):.6f}s "
        f"median={statistics.median(timings):.6f}s "
        f"mean={statistics.mean(timings):.6f}s "
        f"merges_median={statistics.median(merge_counts)}"
    )


if __name__ == "__main__":
    main()
