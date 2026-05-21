import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import puzzle_solver as Solver  # noqa: E402


def build_segments(image_size, piece_size, seed, score_storage):
    rng = np.random.default_rng(seed)
    image = rng.integers(
        0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    return Solver.breakUpImage(
        image,
        piece_size,
        save_segments=False,
        color_type=Solver.ColorType.RGB,
        score_storage=score_storage,
    )


def time_score_run(
        image_size,
        piece_size,
        seed,
        executor_type,
        workers,
        score_storage,
        score_mode):
    segments = build_segments(image_size, piece_size, seed, score_storage)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        Solver.calculateScores(
            segments,
            Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=workers,
            executor_type=executor_type,
        )
    score_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    Solver.finalizeScores(
        segments,
        Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
        score_mode,
    )
    finalize_elapsed = time.perf_counter() - started
    return score_elapsed, finalize_elapsed, len(segments[0].score_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--piece-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--score-storage",
        choices=("dense", "dict"),
        default="dense",
    )
    parser.add_argument(
        "--score-mode",
        choices=("dissimilarity", "reliability"),
        default="dissimilarity",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["serial:1", "thread:4", "process:4"],
        help="Executor runs as executor:workers, for example serial:1 process:8",
    )
    args = parser.parse_args()
    score_mode = Solver.ScoreMode[args.score_mode.upper()]

    segment_count = (args.image_size // args.piece_size) ** 2
    print(f"segments: {segment_count}")
    print(f"score_storage: {args.score_storage}")
    print(f"score_mode: {args.score_mode}")
    for run in args.runs:
        executor_type, workers_text = run.split(":", 1)
        workers = int(workers_text)
        score_elapsed, finalize_elapsed, score_count = time_score_run(
            args.image_size,
            args.piece_size,
            args.seed,
            executor_type,
            workers,
            args.score_storage,
            score_mode,
        )
        print(
            f"{executor_type:7s} workers={workers}: "
            f"score={score_elapsed:.3f}s finalize={finalize_elapsed:.3f}s "
            f"total={score_elapsed + finalize_elapsed:.3f}s scores={score_count}"
        )


if __name__ == "__main__":
    main()
