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


def build_segments(image_size, piece_size, seed):
    rng = np.random.default_rng(seed)
    image = rng.integers(
        0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    return Solver.breakUpImage(
        image,
        piece_size,
        save_segments=False,
        color_type=Solver.ColorType.RGB,
    )


def time_score_run(image_size, piece_size, seed, executor_type, workers):
    segments = build_segments(image_size, piece_size, seed)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        Solver.calculateScores(
            segments,
            Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=workers,
            executor_type=executor_type,
        )
    return time.perf_counter() - started, len(segments[0].score_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--piece-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["serial:1", "thread:4", "process:4"],
        help="Executor runs as executor:workers, for example serial:1 process:8",
    )
    args = parser.parse_args()

    segment_count = (args.image_size // args.piece_size) ** 2
    print(f"segments: {segment_count}")
    for run in args.runs:
        executor_type, workers_text = run.split(":", 1)
        workers = int(workers_text)
        elapsed, score_count = time_score_run(
            args.image_size,
            args.piece_size,
            args.seed,
            executor_type,
            workers,
        )
        print(
            f"{executor_type:7s} workers={workers}: "
            f"{elapsed:.3f}s scores={score_count}"
        )


if __name__ == "__main__":
    main()
