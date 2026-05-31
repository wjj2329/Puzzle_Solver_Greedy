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
    best_frame = None
    best_trimmed = None
    best_key = None
    for row_start, col_start in trimFrameStarts(segment):
        frame, trimmed = trimFrame(segment, row_start, col_start)
        occupied = int(np.count_nonzero(frame))
        key = (occupied, -len(trimmed), -abs(row_start), -abs(col_start))
        if best_key is None or key > best_key:
            best_key = key
            best_frame = frame
            best_trimmed = trimmed

    segment.pic_connection_matrix = best_frame
    segment.binary_connection_matrix = (best_frame != 0).astype(int)
    return best_trimmed


def trimFrame(segment, row_start, col_start):
    frame_height = segment.max_height
    frame_width = segment.max_width
    matrix = segment.pic_connection_matrix
    height, width = matrix.shape
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
    return frame, trimmed


def trimFrameStarts(segment):
    frame_height = segment.max_height
    frame_width = segment.max_width
    height, width = segment.pic_connection_matrix.shape
    row_starts = range(min(0, height - frame_height), max(0, height - frame_height) + 1)
    col_starts = range(min(0, width - frame_width), max(0, width - frame_width) + 1)
    for row_start in row_starts:
        for col_start in col_starts:
            yield row_start, col_start


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


def componentTargetPositions(component, row_offset, col_offset):
    component_matrix = component.pic_connection_matrix
    return {
        (int(row + row_offset), int(col + col_offset))
        for row, col in zip(*np.where(component_matrix != 0))
    }


def componentPlacementScore(frame, component, row_offset, col_offset):
    score_dict = component.score_dict
    total = 0.0
    count = 0
    component_matrix = component.pic_connection_matrix
    height, width = frame.shape
    target_positions = componentTargetPositions(
        component,
        row_offset,
        col_offset,
    )
    for row, col in zip(*np.where(component_matrix != 0)):
        piece = component_matrix[row, col]
        frame_row = row + row_offset
        frame_col = col + col_offset
        for row_delta, col_delta, direction in NEIGHBOR_DIRECTIONS:
            neighbor_row = frame_row + row_delta
            neighbor_col = frame_col + col_delta
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue
            if (neighbor_row, neighbor_col) in target_positions:
                continue
            neighbor = frame[neighbor_row, neighbor_col]
            if neighbor == 0:
                continue
            score = score_dict.get(
                (neighbor.piece_number, direction, piece.piece_number),
                math.inf,
            )
            if math.isinf(score):
                return math.inf, count
            total += score
            count += 1
    if count == 0:
        return math.inf, count
    return total / count, count


def componentOverlap(frame, component, row_offset, col_offset):
    overlap = []
    component_matrix = component.pic_connection_matrix
    for row, col in zip(*np.where(component_matrix != 0)):
        piece = frame[row + row_offset, col + col_offset]
        if piece != 0:
            overlap.append(piece)
    return overlap


def iterComponentPlacements(frame, component, allow_overlap=False):
    component_matrix = component.pic_connection_matrix
    occupied_rows, occupied_cols = np.where(component_matrix != 0)
    if len(occupied_rows) == 0:
        return

    frame_height, frame_width = frame.shape
    min_row = int(occupied_rows.min())
    max_row = int(occupied_rows.max())
    min_col = int(occupied_cols.min())
    max_col = int(occupied_cols.max())

    for row_offset in range(-min_row, frame_height - max_row):
        for col_offset in range(-min_col, frame_width - max_col):
            if allow_overlap:
                yield row_offset, col_offset
                continue
            blocked = False
            for row, col in zip(occupied_rows, occupied_cols):
                if frame[row + row_offset, col + col_offset] != 0:
                    blocked = True
                    break
            if not blocked:
                yield row_offset, col_offset


def placeComponent(frame, component, row_offset, col_offset):
    displaced = []
    component_matrix = component.pic_connection_matrix
    for row, col in zip(*np.where(component_matrix != 0)):
        frame_row = row + row_offset
        frame_col = col + col_offset
        previous = frame[frame_row, frame_col]
        if previous != 0:
            displaced.append(previous)
        frame[frame_row, frame_col] = component_matrix[row, col]
    return displaced


def placeComponentsInFrame(frame, components, allow_overlap=False):
    unplaced = list(components)
    placed = 0
    placed_pieces = 0
    displaced_pieces = []
    total_neighbor_count = 0
    total_score = 0.0
    while unplaced:
        best = None
        for component_index, component in enumerate(unplaced):
            component_size = componentSize(component)
            for row_offset, col_offset in iterComponentPlacements(
                    frame,
                    component,
                    allow_overlap=allow_overlap):
                overlap = componentOverlap(
                    frame,
                    component,
                    row_offset,
                    col_offset,
                )
                net_new_pieces = component_size - len(overlap)
                if net_new_pieces <= 0:
                    continue
                score, neighbor_count = componentPlacementScore(
                    frame,
                    component,
                    row_offset,
                    col_offset,
                )
                if math.isinf(score):
                    continue
                item = (
                    -net_new_pieces,
                    -neighbor_count,
                    score,
                    len(overlap),
                    -component_size,
                    component.piece_number,
                    row_offset,
                    col_offset,
                    component_index,
                )
                if best is None or item < best:
                    best = item
        if best is None:
            break
        (
            _net_new_pieces,
            _neighbor_count,
            score,
            _overlap_count,
            _component_size,
            _piece_number,
            row_offset,
            col_offset,
            component_index,
        ) = best
        neighbor_count = -_neighbor_count
        component = unplaced.pop(component_index)
        displaced_pieces.extend(
            placeComponent(frame, component, row_offset, col_offset)
        )
        placed += 1
        placed_pieces += componentSize(component)
        total_neighbor_count += neighbor_count
        total_score += score * neighbor_count

    return (
        placed,
        placed_pieces,
        unplaced,
        displaced_pieces,
        total_neighbor_count,
        total_score,
    )


def placeComponents(segment, components):
    frame = segment.pic_connection_matrix
    (
        placed,
        _placed_pieces,
        unplaced,
        displaced_pieces,
        _neighbor_count,
        _score,
    ) = placeComponentsInFrame(
        frame,
        components,
    )
    segment.binary_connection_matrix = (frame != 0).astype(int)
    return placed, unplaced, displaced_pieces


def trimToBestFrameWithComponents(segment, components):
    best_frame = None
    best_trimmed = None
    best_unplaced = None
    best_key = None
    for row_start, col_start in trimFrameStarts(segment):
        frame, trimmed = trimFrame(segment, row_start, col_start)
        root_occupied = int(np.count_nonzero(frame))
        (
            placed,
            placed_pieces,
            unplaced,
            displaced,
            neighbor_count,
            score,
        ) = placeComponentsInFrame(
            frame,
            components,
            allow_overlap=True,
        )
        occupied = int(np.count_nonzero(frame))
        key = (
            occupied,
            placed_pieces,
            placed,
            neighbor_count,
            -score,
            root_occupied,
            -len(trimmed),
            -abs(row_start),
            -abs(col_start),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_frame = frame
            best_trimmed = trimmed + displaced
            best_unplaced = unplaced

    segment.pic_connection_matrix = best_frame
    segment.binary_connection_matrix = (best_frame != 0).astype(int)
    placed = len(components) - len(best_unplaced)
    return best_trimmed, placed, best_unplaced


def trimAndFillAssembly(
        segment_list,
        show_progress=True,
        preserve_components=False):
    if not segment_list:
        return None

    root = max(segment_list, key=componentSize)
    leftover_components = []
    for segment in segment_list:
        if segment is root:
            continue
        leftover_components.append(segment)
    if preserve_components:
        trimmed_pieces, placed_components, leftover_components = (
            trimToBestFrameWithComponents(
                root,
                leftover_components,
            )
        )
    else:
        trimmed_pieces = trimToBestFrame(root)
        placed_components = 0
    if preserve_components and leftover_components:
        additional_placed, leftover_components, displaced_pieces = placeComponents(
            root,
            leftover_components,
        )
        placed_components += additional_placed
        trimmed_pieces.extend(displaced_pieces)
    candidates = []
    for component in leftover_components:
        candidates.extend(iterPieces(component))
    candidates.extend(trimmed_pieces)

    if show_progress:
        missing = root.max_height * root.max_width - componentSize(root)
        print(
            "Trim/fill post-process: "
            f"starting with {missing} holes and {len(candidates)} candidates",
            flush=True,
        )
        if preserve_components:
            print(
                "Trim/fill post-process: "
                f"placed {placed_components} leftover components",
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
