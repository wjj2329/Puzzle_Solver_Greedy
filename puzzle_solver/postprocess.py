import math

import numpy as np

from .enums import JoinDirection
from .models import BestConnection


NEIGHBOR_DIRECTIONS = (
    (-1, 0, JoinDirection.DOWN),
    (1, 0, JoinDirection.UP),
    (0, -1, JoinDirection.RIGHT),
    (0, 1, JoinDirection.LEFT),
)


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


def fillScore(frame, row, col, candidate):
    total = 0.0
    count = 0
    score_dict = candidate.score_dict
    for neighbor, direction in adjacentPieces(frame, row, col):
        score = score_dict.get(
            (neighbor.piece_number, direction, candidate.piece_number),
            math.inf,
        )
        if math.isinf(score):
            return math.inf, count
        total += score
        count += 1
    if count == 0:
        return math.inf, count
    return total / count, count


def fillHoles(segment, candidates):
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
    filled = 0
    while remaining:
        best = None
        for row, col in zip(*np.where(frame == 0)):
            for candidate in remaining.values():
                score, neighbor_count = fillScore(frame, row, col, candidate)
                if math.isinf(score):
                    continue
                item = (-neighbor_count, score, candidate.piece_number, row, col, candidate)
                if best is None or item < best:
                    best = item
        if best is None:
            break
        _, _, piece_number, row, col, candidate = best
        frame[row, col] = candidate
        del remaining[piece_number]
        filled += 1

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

    filled = fillHoles(root, candidates)
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
