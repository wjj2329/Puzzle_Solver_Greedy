import heapq
import math
import time

import numpy as np

from .enums import JoinDirection
from .models import BestConnection


NEIGHBOR_DIRECTIONS = (
    (-1, 0, JoinDirection.DOWN),
    (1, 0, JoinDirection.UP),
    (0, -1, JoinDirection.RIGHT),
    (0, 1, JoinDirection.LEFT),
)
FILL_PROGRESS_INTERVAL_SECONDS = 5.0


def componentSize(segment):
    return int(np.count_nonzero(segment.binary_connection_matrix))


def iterPieces(segment):
    seen = set()
    for piece in segment.pic_connection_matrix.flat:
        if piece == 0:
            continue
        if piece.piece_number in seen:
            continue
        seen.add(piece.piece_number)
        yield piece


def trimToBestFrame(segment):
    frame_height = segment.max_height
    frame_width = segment.max_width
    matrix = segment.pic_connection_matrix
    height, width = matrix.shape
    row_starts = range(min(0, height - frame_height), max(0, height - frame_height) + 1)
    col_starts = range(min(0, width - frame_width), max(0, width - frame_width) + 1)

    best_frame = None
    best_trimmed = None
    best_key = None
    for row_start in row_starts:
        for col_start in col_starts:
            frame = np.zeros((frame_height, frame_width), dtype=object)
            trimmed = []
            for row in range(height):
                for col in range(width):
                    piece = matrix[row, col]
                    if piece == 0:
                        continue
                    frame_row = row - row_start
                    frame_col = col - col_start
                    if 0 <= frame_row < frame_height and 0 <= frame_col < frame_width:
                        frame[frame_row, frame_col] = piece
                    else:
                        trimmed.append(piece)
            occupied = int(np.count_nonzero(frame))
            key = (occupied, -len(trimmed), -abs(row_start), -abs(col_start))
            if best_key is None or key > best_key:
                best_key = key
                best_frame = frame
                best_trimmed = trimmed

    segment.pic_connection_matrix = best_frame
    segment.binary_connection_matrix = (best_frame != 0).astype(int)
    return best_trimmed


def adjacentPieces(frame, row, col):
    height, width = frame.shape
    for row_delta, col_delta, direction in NEIGHBOR_DIRECTIONS:
        neighbor_row = row + row_delta
        neighbor_col = col + col_delta
        if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
            continue
        neighbor = frame[neighbor_row, neighbor_col]
        if neighbor != 0:
            yield neighbor, direction


def adjacentEmptyHoles(frame, row, col):
    height, width = frame.shape
    for row_delta, col_delta, _direction in NEIGHBOR_DIRECTIONS:
        neighbor_row = row + row_delta
        neighbor_col = col + col_delta
        if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
            continue
        if frame[neighbor_row, neighbor_col] == 0:
            yield neighbor_row, neighbor_col


def holeNeighborData(frame, row, col):
    return tuple(
        (neighbor.piece_number, direction)
        for neighbor, direction in adjacentPieces(frame, row, col)
    )


def fillScoreForNeighbors(neighbor_data, score_dict, candidate_piece_number):
    total = 0.0
    count = len(neighbor_data)
    if count == 0:
        return math.inf, count
    for neighbor_piece_number, direction in neighbor_data:
        score = score_dict.get(
            (neighbor_piece_number, direction, candidate_piece_number),
            math.inf,
        )
        if math.isinf(score):
            return math.inf, count
        total += score
    return total / count, count


def fillScore(frame, row, col, candidate):
    return fillScoreForNeighbors(
        holeNeighborData(frame, row, col),
        candidate.score_dict,
        candidate.piece_number,
    )


def pushHoleScores(heap, frame, row, col, remaining, hole_versions):
    if frame[row, col] != 0:
        return
    neighbor_data = holeNeighborData(frame, row, col)
    if not neighbor_data:
        return
    version = hole_versions[row, col]
    for candidate in remaining.values():
        score, neighbor_count = fillScoreForNeighbors(
            neighbor_data,
            candidate.score_dict,
            candidate.piece_number,
        )
        if math.isinf(score):
            continue
        heapq.heappush(
            heap,
            (
                -neighbor_count,
                score,
                candidate.piece_number,
                row,
                col,
                version,
            ),
        )


def fillHoles(
        segment,
        candidates,
        show_progress=False,
        progress_interval=FILL_PROGRESS_INTERVAL_SECONDS):
    frame = segment.pic_connection_matrix
    remaining = {
        candidate.piece_number: candidate
        for candidate in candidates
        if candidate.piece_number not in {
            piece.piece_number
            for piece in frame.flat
            if piece != 0
        }
    }
    holes = {
        (int(row), int(col))
        for row, col in zip(*np.where(frame == 0))
    }
    hole_versions = {
        hole: 0
        for hole in holes
    }
    heap = []
    for row, col in holes:
        pushHoleScores(heap, frame, row, col, remaining, hole_versions)

    filled = 0
    next_progress = time.perf_counter() + progress_interval
    while remaining:
        best = None
        while heap:
            item = heapq.heappop(heap)
            _neighbor_count, _score, piece_number, row, col, version = item
            if piece_number not in remaining:
                continue
            if frame[row, col] != 0:
                continue
            if hole_versions.get((row, col)) != version:
                continue
            best = item
            break
        if best is None:
            break
        _, _, piece_number, row, col, _version = best
        candidate = remaining[piece_number]
        frame[row, col] = candidate
        holes.discard((row, col))
        del remaining[piece_number]
        filled += 1

        for neighbor_row, neighbor_col in adjacentEmptyHoles(frame, row, col):
            if (neighbor_row, neighbor_col) not in holes:
                continue
            hole_versions[neighbor_row, neighbor_col] += 1
            pushHoleScores(
                heap,
                frame,
                neighbor_row,
                neighbor_col,
                remaining,
                hole_versions,
            )

        if show_progress and time.perf_counter() >= next_progress:
            print(
                "Trim/fill progress: "
                f"filled {filled}, {len(holes)} holes remain, "
                f"{len(remaining)} candidates remain",
                flush=True,
            )
            next_progress = time.perf_counter() + progress_interval

    segment.binary_connection_matrix = (frame != 0).astype(int)
    return filled


def trimAndFillAssembly(segment_list, show_progress=True):
    if not segment_list:
        return None

    root = max(segment_list, key=componentSize)
    candidates = []
    for segment in segment_list:
        if segment is root:
            continue
        candidates.extend(iterPieces(segment))
    candidates.extend(trimToBestFrame(root))

    if show_progress:
        missing = root.max_height * root.max_width - componentSize(root)
        print(
            "Trim/fill post-process: "
            f"starting with {missing} holes and {len(candidates)} candidates",
            flush=True,
        )
    filled = fillHoles(root, candidates, show_progress=show_progress)
    segment_list[:] = [root]

    if show_progress:
        missing = root.max_height * root.max_width - componentSize(root)
        print(
            "Trim/fill post-process: "
            f"filled {filled} holes, {missing} holes remain"
        )
    return BestConnection(
        own_segment=root,
        pic_connection_matrix=root.pic_connection_matrix,
        binary_connection_matrix=root.binary_connection_matrix,
    )
