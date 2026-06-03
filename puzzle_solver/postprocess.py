import heapq
import math
import time

import numpy as np

from .enums import JoinDirection, OPPOSITE_DIRECTIONS
from .models import BestConnection


NEIGHBOR_DIRECTIONS = (
    (-1, 0, JoinDirection.DOWN),
    (1, 0, JoinDirection.UP),
    (0, -1, JoinDirection.RIGHT),
    (0, 1, JoinDirection.LEFT),
)
BASE_TO_COMPONENT_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
SPLIT_SEAM_DIRECTIONS = (
    (0, 1, JoinDirection.RIGHT),
    (1, 0, JoinDirection.DOWN),
)
CONSENSUS_SHIFT_PROGRESS_INTERVAL_SECONDS = 5.0
FILL_PROGRESS_INTERVAL_SECONDS = 5.0
CONSERVATIVE_FILL_SCORE_QUANTILE = 0.90
CONSERVATIVE_FILL_SCORE_MULTIPLIER = 1.10
EDGE_PRESERVING_MIN_RETAINED_RATIO = 0.98
EDGE_PRESERVING_MAX_SCORE_MULTIPLIER = 1.10
ENDGAME_MAX_FRAME_OVERFLOW = 96


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


def frameAdjacencyScoreStats(frame):
    scores = []
    adjacent_count = 0
    height, width = frame.shape
    for row in range(height):
        for col in range(width):
            piece = frame[row, col]
            if piece == 0:
                continue
            if col + 1 < width and frame[row, col + 1] != 0:
                adjacent_count += 1
                score = piece.score_dict.get(
                    (
                        piece.piece_number,
                        JoinDirection.RIGHT,
                        frame[row, col + 1].piece_number,
                    ),
                    math.inf,
                )
                if not math.isinf(score):
                    scores.append(float(score))
            if row + 1 < height and frame[row + 1, col] != 0:
                adjacent_count += 1
                score = piece.score_dict.get(
                    (
                        piece.piece_number,
                        JoinDirection.DOWN,
                        frame[row + 1, col].piece_number,
                    ),
                    math.inf,
                )
                if not math.isinf(score):
                    scores.append(float(score))
    if not scores:
        return adjacent_count, math.inf, 0
    return adjacent_count, sum(scores) / len(scores), len(scores)


def segmentListAdjacencyScoreStats(segment_list):
    adjacent_count = 0
    score_count = 0
    total_score = 0.0
    for segment in segment_list:
        segment_adjacent_count, average_score, segment_score_count = (
            frameAdjacencyScoreStats(segment.pic_connection_matrix)
        )
        adjacent_count += segment_adjacent_count
        if segment_score_count == 0:
            continue
        score_count += segment_score_count
        total_score += average_score * segment_score_count
    if score_count == 0:
        return adjacent_count, math.inf, 0
    return adjacent_count, total_score / score_count, score_count


def snapshotSegments(segment_list):
    return (
        list(segment_list),
        [
            (
                segment,
                np.array(segment.pic_connection_matrix, copy=True),
                np.array(segment.binary_connection_matrix, copy=True),
            )
            for segment in segment_list
        ],
    )


def restoreSegments(segment_list, snapshot):
    original_segments, segment_states = snapshot
    for segment, pic_matrix, binary_matrix in segment_states:
        segment.pic_connection_matrix = pic_matrix
        segment.binary_connection_matrix = binary_matrix
    segment_list[:] = original_segments


def shouldKeepOriginalAssembly(original_stats, final_stats):
    original_adjacent_count, original_average_score, _original_score_count = (
        original_stats
    )
    final_adjacent_count, final_average_score, _final_score_count = final_stats
    if original_adjacent_count == 0:
        return False
    minimum_retained = (
        original_adjacent_count * EDGE_PRESERVING_MIN_RETAINED_RATIO
    )
    if final_adjacent_count < minimum_retained:
        return True
    if (
            final_adjacent_count <= original_adjacent_count
            and not math.isinf(original_average_score)
            and final_average_score
            > original_average_score * EDGE_PRESERVING_MAX_SCORE_MULTIPLIER):
        return True
    return False


def conservativeFillScoreLimit(frame):
    scores = []
    height, width = frame.shape
    for row in range(height):
        for col in range(width):
            piece = frame[row, col]
            if piece == 0:
                continue
            if col + 1 < width and frame[row, col + 1] != 0:
                score = piece.score_dict.get(
                    (
                        piece.piece_number,
                        JoinDirection.RIGHT,
                        frame[row, col + 1].piece_number,
                    ),
                    math.inf,
                )
                if not math.isinf(score):
                    scores.append(float(score))
            if row + 1 < height and frame[row + 1, col] != 0:
                score = piece.score_dict.get(
                    (
                        piece.piece_number,
                        JoinDirection.DOWN,
                        frame[row + 1, col].piece_number,
                    ),
                    math.inf,
                )
                if not math.isinf(score):
                    scores.append(float(score))
    if not scores:
        return None
    return (
        float(np.quantile(scores, CONSERVATIVE_FILL_SCORE_QUANTILE))
        * CONSERVATIVE_FILL_SCORE_MULTIPLIER
    )


def bestDirectionalScore(score_dict, piece_number, direction):
    if hasattr(score_dict, "bestScoreForDirection"):
        return score_dict.bestScoreForDirection(piece_number, direction)

    best = math.inf
    for own_number, score_direction, _join_number in score_dict:
        if own_number != piece_number or score_direction != direction:
            continue
        score = score_dict[own_number, score_direction, _join_number]
        if isinstance(score, tuple):
            score = score[0]
        if score < best:
            best = score
    return best


def frameBorderScore(frame):
    total = 0.0
    count = 0
    height, width = frame.shape
    for row in range(height):
        for col in range(width):
            piece = frame[row, col]
            if piece == 0:
                continue
            directions = []
            if row == 0:
                directions.append(JoinDirection.UP)
            if row == height - 1:
                directions.append(JoinDirection.DOWN)
            if col == 0:
                directions.append(JoinDirection.LEFT)
            if col == width - 1:
                directions.append(JoinDirection.RIGHT)
            for direction in directions:
                score = bestDirectionalScore(
                    piece.score_dict,
                    piece.piece_number,
                    direction,
                )
                if math.isinf(score):
                    continue
                total += score
                count += 1
    return total, count


def trimToBestFrame(
        segment,
        score_tiebreak=False,
        border_tiebreak=False,
        edge_preserving=False):
    best_frame = None
    best_trimmed = None
    best_key = None
    for row_start, col_start in trimFrameStarts(segment):
        frame, trimmed = trimFrame(segment, row_start, col_start)
        occupied = int(np.count_nonzero(frame))
        if edge_preserving:
            adjacent_count, average_score, score_count = frameAdjacencyScoreStats(
                frame,
            )
            border_score, border_count = (
                frameBorderScore(frame) if border_tiebreak else (0.0, 0)
            )
            key = (
                adjacent_count,
                score_count,
                -average_score,
                occupied,
                border_score,
                border_count,
                -len(trimmed),
                -abs(row_start),
                -abs(col_start),
            )
        elif score_tiebreak or border_tiebreak:
            adjacent_count, average_score, score_count = frameAdjacencyScoreStats(
                frame,
            )
            border_score, border_count = (
                frameBorderScore(frame) if border_tiebreak else (0.0, 0)
            )
            key = (
                occupied,
                border_score,
                border_count,
                adjacent_count,
                score_count,
                -average_score,
                -len(trimmed),
                -abs(row_start),
                -abs(col_start),
            )
        else:
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


def seamRepairScore(piece, neighbor, direction):
    opposite_direction = OPPOSITE_DIRECTIONS[direction]
    scores = []
    for own_piece, own_direction, join_piece in (
            (piece, direction, neighbor),
            (neighbor, opposite_direction, piece)):
        score = own_piece.score_dict.get(
            (
                own_piece.piece_number,
                own_direction,
                join_piece.piece_number,
            ),
            math.inf,
        )
        if not math.isinf(score):
            scores.append(float(score))
    if not scores:
        return math.inf
    return max(scores)


def scoreAsFloat(score):
    if isinstance(score, tuple):
        return float(score[0])
    return float(score)


def topCandidatesForDirection(score_dict, piece_number, direction, limit, cache):
    key = (piece_number, direction, limit)
    if key in cache:
        return cache[key]

    scalar_values = getattr(score_dict, "scalarScoreValues", lambda: None)()
    if scalar_values is not None:
        direction_index = list(JoinDirection).index(direction)
        scores = np.asarray(
            scalar_values[piece_number, direction_index, :],
            dtype=np.float64,
        ).copy()
        if scores.size == 0:
            cache[key] = frozenset()
            return cache[key]
        scores[0] = np.inf
        scores[piece_number] = np.inf
        scores[np.isnan(scores)] = np.inf
        finite = np.isfinite(scores)
        if not np.any(finite):
            cache[key] = frozenset()
            return cache[key]
        candidate_limit = min(limit, int(np.count_nonzero(finite)))
        candidate_indices = np.argpartition(
            scores,
            candidate_limit - 1,
        )[:candidate_limit]
        candidates = frozenset(
            int(candidate)
            for candidate in candidate_indices
            if np.isfinite(scores[candidate])
        )
        cache[key] = candidates
        return candidates

    candidates = []
    for own_piece, score_direction, join_piece in score_dict:
        if own_piece != piece_number or score_direction != direction:
            continue
        if join_piece == piece_number:
            continue
        candidates.append((
            scoreAsFloat(score_dict[own_piece, score_direction, join_piece]),
            join_piece,
        ))
    cache[key] = frozenset(
        join_piece
        for _score, join_piece in heapq.nsmallest(limit, candidates)
    )
    return cache[key]


def isMutualTopCandidate(piece, neighbor, direction, top_k, cache):
    opposite_direction = OPPOSITE_DIRECTIONS[direction]
    score_dict = piece.score_dict
    return (
        neighbor.piece_number
        in topCandidatesForDirection(
            score_dict,
            piece.piece_number,
            direction,
            top_k,
            cache,
        )
        and piece.piece_number
        in topCandidatesForDirection(
            score_dict,
            neighbor.piece_number,
            opposite_direction,
            top_k,
            cache,
        )
    )


def seamLocalSupport(frame, row, col, direction, top_k, cache):
    height, width = frame.shape
    support = 0

    def has_mutual_seam(first_row, first_col, seam_direction):
        row_delta, col_delta = {
            JoinDirection.RIGHT: (0, 1),
            JoinDirection.DOWN: (1, 0),
        }[seam_direction]
        second_row = first_row + row_delta
        second_col = first_col + col_delta
        if not (
                0 <= first_row < height
                and 0 <= first_col < width
                and 0 <= second_row < height
                and 0 <= second_col < width):
            return False
        first_piece = frame[first_row, first_col]
        second_piece = frame[second_row, second_col]
        if first_piece == 0 or second_piece == 0:
            return False
        return isMutualTopCandidate(
            first_piece,
            second_piece,
            seam_direction,
            top_k,
            cache,
        )

    if direction == JoinDirection.RIGHT:
        if (
                has_mutual_seam(row - 1, col, JoinDirection.RIGHT)
                and has_mutual_seam(row - 1, col, JoinDirection.DOWN)
                and has_mutual_seam(row - 1, col + 1, JoinDirection.DOWN)):
            support += 1
        if (
                has_mutual_seam(row + 1, col, JoinDirection.RIGHT)
                and has_mutual_seam(row, col, JoinDirection.DOWN)
                and has_mutual_seam(row, col + 1, JoinDirection.DOWN)):
            support += 1
    elif direction == JoinDirection.DOWN:
        if (
                has_mutual_seam(row, col - 1, JoinDirection.DOWN)
                and has_mutual_seam(row, col - 1, JoinDirection.RIGHT)
                and has_mutual_seam(row + 1, col - 1, JoinDirection.RIGHT)):
            support += 1
        if (
                has_mutual_seam(row, col + 1, JoinDirection.DOWN)
                and has_mutual_seam(row, col, JoinDirection.RIGHT)
                and has_mutual_seam(row + 1, col, JoinDirection.RIGHT)):
            support += 1
    return support


def splitSegmentByConsensus(
        segment,
        top_k,
        min_local_support,
        next_component_id):
    frame = segment.pic_connection_matrix
    positions = {
        (int(row), int(col))
        for row, col in zip(*np.where(frame != 0))
    }
    if len(positions) <= 1:
        segment._consensus_origin = (0, 0)
        return [segment], 0, next_component_id

    adjacency = {
        position: []
        for position in positions
    }
    cut_count = 0
    cache = {}
    height, width = frame.shape
    for row, col in positions:
        piece = frame[row, col]
        for row_delta, col_delta, direction in SPLIT_SEAM_DIRECTIONS:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue
            neighbor_position = (neighbor_row, neighbor_col)
            if neighbor_position not in positions:
                continue
            neighbor = frame[neighbor_position]
            keep_seam = isMutualTopCandidate(
                piece,
                neighbor,
                direction,
                top_k,
                cache,
            )
            if keep_seam and min_local_support > 0:
                keep_seam = (
                    seamLocalSupport(frame, row, col, direction, top_k, cache)
                    >= min_local_support
                )
            if not keep_seam:
                cut_count += 1
                continue
            adjacency[(row, col)].append(neighbor_position)
            adjacency[neighbor_position].append((row, col))

    components = []
    remaining = set(positions)
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            position = stack.pop()
            for neighbor_position in adjacency[position]:
                if neighbor_position not in remaining:
                    continue
                remaining.remove(neighbor_position)
                component.add(neighbor_position)
                stack.append(neighbor_position)
        components.append(component)

    if len(components) <= 1:
        segment._consensus_origin = (0, 0)
        return [segment], cut_count, next_component_id

    components.sort(key=lambda component: min(component))
    split_segments = []
    for component in components:
        min_row = min(row for row, _col in component)
        max_row = max(row for row, _col in component)
        min_col = min(col for _row, col in component)
        max_col = max(col for _row, col in component)
        split_frame = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1),
            dtype=object,
        )
        split_binary = np.zeros(split_frame.shape)
        for row, col in component:
            split_row = row - min_row
            split_col = col - min_col
            split_frame[split_row, split_col] = frame[row, col]
            split_binary[split_row, split_col] = 1

        base_position = min(component)
        split_segment = frame[base_position]
        split_segment.pic_connection_matrix = split_frame
        split_segment.binary_connection_matrix = split_binary
        split_segment.component_id = next_component_id
        split_segment.enforce_frame_bounds = getattr(
            segment,
            "enforce_frame_bounds",
            True,
        )
        split_segment._kruskal_component_data = None
        split_segment._consensus_origin = (min_row, min_col)
        next_component_id += 1
        split_segments.append(split_segment)

    return split_segments, cut_count, next_component_id


def splitSegmentByBadJoins(segment, max_score, next_component_id):
    frame = segment.pic_connection_matrix
    positions = {
        (int(row), int(col))
        for row, col in zip(*np.where(frame != 0))
    }
    if len(positions) <= 1:
        return [segment], 0, next_component_id

    adjacency = {
        position: []
        for position in positions
    }
    cut_count = 0
    height, width = frame.shape
    for row, col in positions:
        piece = frame[row, col]
        for row_delta, col_delta, direction in SPLIT_SEAM_DIRECTIONS:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue
            neighbor_position = (neighbor_row, neighbor_col)
            if neighbor_position not in positions:
                continue
            neighbor = frame[neighbor_position]
            score = seamRepairScore(piece, neighbor, direction)
            if score > max_score:
                cut_count += 1
                continue
            adjacency[(row, col)].append(neighbor_position)
            adjacency[neighbor_position].append((row, col))

    if cut_count == 0:
        return [segment], 0, next_component_id

    components = []
    remaining = set(positions)
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            position = stack.pop()
            for neighbor_position in adjacency[position]:
                if neighbor_position not in remaining:
                    continue
                remaining.remove(neighbor_position)
                component.add(neighbor_position)
                stack.append(neighbor_position)
        components.append(component)

    if len(components) <= 1:
        return [segment], cut_count, next_component_id

    split_segments = []
    for component in components:
        min_row = min(row for row, _col in component)
        max_row = max(row for row, _col in component)
        min_col = min(col for _row, col in component)
        max_col = max(col for _row, col in component)
        split_frame = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1),
            dtype=object,
        )
        split_binary = np.zeros(split_frame.shape)
        for row, col in component:
            split_row = row - min_row
            split_col = col - min_col
            split_frame[split_row, split_col] = frame[row, col]
            split_binary[split_row, split_col] = 1

        base_position = min(component)
        split_segment = frame[base_position]
        split_segment.pic_connection_matrix = split_frame
        split_segment.binary_connection_matrix = split_binary
        split_segment.component_id = next_component_id
        split_segment.enforce_frame_bounds = getattr(
            segment,
            "enforce_frame_bounds",
            True,
        )
        split_segment._kruskal_component_data = None
        next_component_id += 1
        split_segments.append(split_segment)

    return split_segments, cut_count, next_component_id


def splitBadJoinComponents(segment_list, max_score):
    if not segment_list:
        return {
            "before": 0,
            "after": 0,
            "cut_edges": 0,
            "split_components": 0,
        }
    next_component_id = max(segment.component_id for segment in segment_list) + 1
    repaired_segments = []
    cut_edges = 0
    split_components = 0
    for segment in segment_list:
        split_segments, segment_cut_edges, next_component_id = (
            splitSegmentByBadJoins(
                segment,
                max_score,
                next_component_id,
            )
        )
        cut_edges += segment_cut_edges
        if len(split_segments) > 1:
            split_components += 1
        repaired_segments.extend(split_segments)

    before = len(segment_list)
    segment_list[:] = repaired_segments
    return {
        "before": before,
        "after": len(segment_list),
        "cut_edges": cut_edges,
        "split_components": split_components,
    }


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
        progress_interval=FILL_PROGRESS_INTERVAL_SECONDS,
        max_score=None,
        min_neighbor_count=1):
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
            neighbor_count = -_neighbor_count
            if neighbor_count < min_neighbor_count:
                continue
            if max_score is not None and item[1] > max_score:
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


def occupiedPiecePositions(matrix):
    for row, col in zip(*np.where(matrix != 0)):
        yield int(row), int(col), matrix[row, col]


def boundaryPiecePositions(matrix):
    height, width = matrix.shape
    for row, col, _piece in occupiedPiecePositions(matrix):
        for row_delta, col_delta, _direction in NEIGHBOR_DIRECTIONS:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                yield row, col
                break
            if matrix[neighbor_row, neighbor_col] == 0:
                yield row, col
                break


def endgameSegmentData(segment):
    positions = tuple(occupiedPiecePositions(segment.pic_connection_matrix))
    rows = [row for row, _col, _piece in positions]
    cols = [col for _row, col, _piece in positions]
    return {
        "segment": segment,
        "positions": positions,
        "boundary": tuple(boundaryPiecePositions(segment.pic_connection_matrix)),
        "lookup": {
            (row, col): piece
            for row, col, piece in positions
        },
        "stats": frameAdjacencyScoreStats(segment.pic_connection_matrix),
        "min_row": min(rows),
        "max_row": max(rows),
        "min_col": min(cols),
        "max_col": max(cols),
    }


def componentBoundaryOffsetsForData(base_data, component_data):
    offsets = set()
    for base_row, base_col in base_data["boundary"]:
        for component_row, component_col in component_data["boundary"]:
            for row_delta, col_delta in BASE_TO_COMPONENT_DIRECTIONS:
                offsets.add((
                    base_row + row_delta - component_row,
                    base_col + col_delta - component_col,
                ))
    return offsets


def componentBoundaryOffsets(base, component):
    return componentBoundaryOffsetsForData(
        endgameSegmentData(base),
        endgameSegmentData(component),
    )


def componentPlacementScoreAgainstMap(base_lookup, component, row_offset, col_offset):
    score_dict = component.score_dict
    total = 0.0
    count = 0
    for row, col, piece in occupiedPiecePositions(component.pic_connection_matrix):
        frame_row = row + row_offset
        frame_col = col + col_offset
        for row_delta, col_delta, direction in NEIGHBOR_DIRECTIONS:
            neighbor = base_lookup.get((
                frame_row + row_delta,
                frame_col + col_delta,
            ))
            if neighbor is None:
                continue
            score = score_dict.get(
                (neighbor.piece_number, direction, piece.piece_number),
                math.inf,
            )
            if math.isinf(score):
                return math.inf, count, total
            total += score
            count += 1
    if count == 0:
        return math.inf, count, total
    return total / count, count, total


def endgameBoundingBox(base_positions, component_positions, row_offset, col_offset):
    rows = [row for row, _col, _piece in base_positions]
    cols = [col for _row, col, _piece in base_positions]
    rows.extend(row + row_offset for row, _col, _piece in component_positions)
    cols.extend(col + col_offset for _row, col, _piece in component_positions)
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)
    return min_row, max_row, min_col, max_col


def endgameBoundingBoxForData(base_data, component_data, row_offset, col_offset):
    min_row = min(base_data["min_row"], component_data["min_row"] + row_offset)
    max_row = max(base_data["max_row"], component_data["max_row"] + row_offset)
    min_col = min(base_data["min_col"], component_data["min_col"] + col_offset)
    max_col = max(base_data["max_col"], component_data["max_col"] + col_offset)
    return min_row, max_row, min_col, max_col


def endgameMergeScore(base_data, component_data, row_offset, col_offset):
    base = base_data["segment"]
    component = component_data["segment"]
    base_positions = base_data["positions"]
    component_positions = component_data["positions"]
    base_lookup = base_data["lookup"]
    min_row, max_row, min_col, max_col = endgameBoundingBoxForData(
        base_data,
        component_data,
        row_offset,
        col_offset,
    )
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    overflow = (
        max(0, height - base.max_height)
        + max(0, width - base.max_width)
    )
    if overflow > ENDGAME_MAX_FRAME_OVERFLOW:
        return None

    for row, col, _piece in component_positions:
        if (row + row_offset, col + col_offset) in base_lookup:
            return None

    cross_average, cross_count, cross_total = componentPlacementScoreAgainstMap(
        base_lookup,
        component,
        row_offset,
        col_offset,
    )
    if math.isinf(cross_average) or cross_count == 0:
        return None

    area = height * width
    base_adjacent, base_average, base_score_count = base_data["stats"]
    component_adjacent, component_average, component_score_count = (
        component_data["stats"]
    )
    combined_score_count = cross_count
    combined_total = cross_total
    if base_score_count:
        combined_score_count += base_score_count
        combined_total += base_average * base_score_count
    if component_score_count:
        combined_score_count += component_score_count
        combined_total += component_average * component_score_count
    if combined_score_count == 0:
        combined_average = math.inf
    else:
        combined_average = combined_total / combined_score_count
    combined_adjacent = base_adjacent + component_adjacent + cross_count
    return (
        cross_count,
        -overflow,
        -cross_average,
        combined_adjacent,
        combined_score_count,
        -combined_average,
        -area,
        -abs(row_offset),
        -abs(col_offset),
    )


def mergeComponentIntoBase(base, component, row_offset, col_offset):
    base_positions = tuple(occupiedPiecePositions(base.pic_connection_matrix))
    component_positions = tuple(occupiedPiecePositions(component.pic_connection_matrix))
    min_row, max_row, min_col, max_col = endgameBoundingBox(
        base_positions,
        component_positions,
        row_offset,
        col_offset,
    )
    frame = np.zeros(
        (max_row - min_row + 1, max_col - min_col + 1),
        dtype=object,
    )
    for row, col, piece in base_positions:
        frame[row - min_row, col - min_col] = piece
    for row, col, piece in component_positions:
        frame[
            row + row_offset - min_row,
            col + col_offset - min_col,
        ] = piece
    base.pic_connection_matrix = frame
    base.binary_connection_matrix = (frame != 0).astype(int)
    base._kruskal_component_data = None


def findBestEndgameMerge(segment_list):
    data_by_segment = {
        segment: endgameSegmentData(segment)
        for segment in segment_list
    }
    best = None
    for base in segment_list:
        base_data = data_by_segment[base]
        for component in segment_list:
            if component is base:
                continue
            component_data = data_by_segment[component]
            for row_offset, col_offset in componentBoundaryOffsetsForData(
                    base_data,
                    component_data,
            ):
                score = endgameMergeScore(
                    base_data,
                    component_data,
                    row_offset,
                    col_offset,
                )
                if score is None:
                    continue
                key = (
                    score,
                    -componentSize(base),
                    -componentSize(component),
                    -base.piece_number,
                    -component.piece_number,
                )
                if best is None or key > best[0]:
                    best = (key, base, component, row_offset, col_offset)
    return best


def connectEndgameComponents(segment_list, show_progress=True, max_components=4):
    if len(segment_list) <= 1 or len(segment_list) > max_components:
        return 0

    merged = 0
    while len(segment_list) > 1:
        best = findBestEndgameMerge(segment_list)
        if best is None:
            break
        _key, base, component, row_offset, col_offset = best
        mergeComponentIntoBase(base, component, row_offset, col_offset)
        segment_list.remove(component)
        merged += 1

    if show_progress:
        print(
            f"Endgame search: merged {merged} leftover components",
            flush=True,
        )
    return merged


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


def placeComponentsInFrameConstrained(
        frame,
        components,
        min_neighbor_count=2,
        max_score=None):
    unplaced = list(components)
    placed = 0
    placed_pieces = 0
    total_neighbor_count = 0
    total_score = 0.0
    while unplaced:
        best = None
        for component_index, component in enumerate(unplaced):
            component_size = componentSize(component)
            for row_offset, col_offset in iterComponentPlacements(frame, component):
                score, neighbor_count = componentPlacementScore(
                    frame,
                    component,
                    row_offset,
                    col_offset,
                )
                if math.isinf(score):
                    continue
                if neighbor_count < min_neighbor_count:
                    continue
                if max_score is not None and score > max_score:
                    continue
                item = (
                    -neighbor_count,
                    score,
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
            _neighbor_count,
            score,
            _component_size,
            _piece_number,
            row_offset,
            col_offset,
            component_index,
        ) = best
        neighbor_count = -_neighbor_count
        component = unplaced.pop(component_index)
        placeComponent(frame, component, row_offset, col_offset)
        placed += 1
        placed_pieces += componentSize(component)
        total_neighbor_count += neighbor_count
        total_score += score * neighbor_count

    return placed, placed_pieces, unplaced, total_neighbor_count, total_score


def placeComponentsInFrameConstrainedLargeFirst(
        frame,
        components,
        min_neighbor_count=2,
        max_score=None):
    unplaced = list(components)
    placed = 0
    placed_pieces = 0
    total_neighbor_count = 0
    total_score = 0.0
    while unplaced:
        best = None
        for component_index, component in enumerate(unplaced):
            component_size = componentSize(component)
            for row_offset, col_offset in iterComponentPlacements(frame, component):
                score, neighbor_count = componentPlacementScore(
                    frame,
                    component,
                    row_offset,
                    col_offset,
                )
                if math.isinf(score):
                    continue
                if neighbor_count < min_neighbor_count:
                    continue
                if max_score is not None and score > max_score:
                    continue
                item = (
                    -component_size,
                    score,
                    -neighbor_count,
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
            _component_size,
            score,
            _neighbor_count,
            _piece_number,
            row_offset,
            col_offset,
            component_index,
        ) = best
        neighbor_count = -_neighbor_count
        component = unplaced.pop(component_index)
        placeComponent(frame, component, row_offset, col_offset)
        placed += 1
        placed_pieces += componentSize(component)
        total_neighbor_count += neighbor_count
        total_score += score * neighbor_count

    return placed, placed_pieces, unplaced, total_neighbor_count, total_score


def repairFrameStarts(segment):
    frame_height = segment.max_height
    frame_width = segment.max_width
    height, width = segment.pic_connection_matrix.shape
    if height <= frame_height:
        row_starts = range(-(frame_height - height), 1)
    else:
        row_starts = range(0, height - frame_height + 1)
    if width <= frame_width:
        col_starts = range(-(frame_width - width), 1)
    else:
        col_starts = range(0, width - frame_width + 1)
    for row_start in row_starts:
        for col_start in col_starts:
            yield row_start, col_start


def placeRepairedComponentsInFrame(
        segment_list,
        min_neighbor_count=2,
        max_score=None,
        large_first=False):
    if len(segment_list) <= 1:
        return {
            "before": len(segment_list),
            "after": len(segment_list),
            "placed_components": 0,
            "placed_pieces": 0,
            "neighbor_count": 0,
            "average_score": math.inf,
        }

    root = max(segment_list, key=componentSize)
    components = [
        component
        for component in segment_list
        if component is not root
    ]
    best = None
    for row_start, col_start in repairFrameStarts(root):
        frame, trimmed = trimFrame(root, row_start, col_start)
        root_preserved_pieces = componentSize(root) - len(trimmed)
        place_components = (
            placeComponentsInFrameConstrainedLargeFirst
            if large_first
            else placeComponentsInFrameConstrained
        )
        (
            placed,
            placed_pieces,
            unplaced,
            neighbor_count,
            placement_score,
        ) = place_components(
            frame,
            components,
            min_neighbor_count=min_neighbor_count,
            max_score=max_score,
        )
        occupied = int(np.count_nonzero(frame))
        average_score_value = averageScore(placement_score, neighbor_count)
        key = (
            placed_pieces,
            placed,
            neighbor_count,
            -average_score_value,
            root_preserved_pieces,
            occupied,
            -len(trimmed),
            -abs(row_start),
            -abs(col_start),
            -root.piece_number,
        )
        if best is None or key > best[0]:
            best = (
                key,
                frame,
                trimmed,
                placed,
                placed_pieces,
                unplaced,
                neighbor_count,
                average_score_value,
            )

    if best is None:
        return {
            "before": len(segment_list),
            "after": len(segment_list),
            "placed_components": 0,
            "placed_pieces": 0,
            "neighbor_count": 0,
            "average_score": math.inf,
        }

    (
        _key,
        frame,
        _trimmed,
        placed,
        placed_pieces,
        unplaced,
        neighbor_count,
        average_score_value,
    ) = best
    root.pic_connection_matrix = frame
    root.binary_connection_matrix = (frame != 0).astype(int)
    root._kruskal_component_data = None
    before = len(segment_list)
    segment_list[:] = [root] + unplaced
    return {
        "before": before,
        "after": len(segment_list),
        "placed_components": placed,
        "placed_pieces": placed_pieces,
        "neighbor_count": neighbor_count,
        "average_score": average_score_value,
    }


def placeConsensusComponentsInFrame(
        segment_list,
        min_neighbor_count=2,
        max_score=None):
    for segment in segment_list:
        stripEmptyBorder(segment)
    return placeRepairedComponentsInFrame(
        segment_list,
        min_neighbor_count=min_neighbor_count,
        max_score=max_score,
    )


def componentOrigin(component):
    return getattr(component, "_consensus_origin", (0, 0))


def placeComponentAtOrigin(frame, component):
    row_offset, col_offset = componentOrigin(component)
    component_matrix = component.pic_connection_matrix
    height, width = frame.shape
    for row, col in zip(*np.where(component_matrix != 0)):
        frame_row = row + row_offset
        frame_col = col + col_offset
        if not (0 <= frame_row < height and 0 <= frame_col < width):
            return False
        if frame[frame_row, frame_col] != 0:
            return False
    placeComponent(frame, component, row_offset, col_offset)
    return True


def iterShiftedComponentPlacements(frame, component, max_shift):
    origin_row, origin_col = componentOrigin(component)
    component_matrix = component.pic_connection_matrix
    occupied_rows, occupied_cols = np.where(component_matrix != 0)
    if len(occupied_rows) == 0:
        return

    frame_height, frame_width = frame.shape
    min_row = int(occupied_rows.min())
    max_row = int(occupied_rows.max())
    min_col = int(occupied_cols.min())
    max_col = int(occupied_cols.max())
    for row_offset in range(origin_row - max_shift, origin_row + max_shift + 1):
        if row_offset + min_row < 0 or row_offset + max_row >= frame_height:
            continue
        for col_offset in range(
                origin_col - max_shift,
                origin_col + max_shift + 1):
            if col_offset + min_col < 0 or col_offset + max_col >= frame_width:
                continue
            blocked = False
            for row, col in zip(occupied_rows, occupied_cols):
                if frame[row + row_offset, col + col_offset] != 0:
                    blocked = True
                    break
            if not blocked:
                yield row_offset, col_offset


def placeConsensusComponentsByShift(
        components,
        min_neighbor_count=3,
        max_shift=1,
        max_score=None):
    if not components:
        return None, [], {
            "placed_components": 0,
            "placed_pieces": 0,
            "neighbor_count": 0,
            "average_score": math.inf,
        }

    root = max(components, key=componentSize)
    frame = np.zeros((root.max_height, root.max_width), dtype=object)
    if not placeComponentAtOrigin(frame, root):
        return root, [component for component in components if component is not root], {
            "placed_components": 0,
            "placed_pieces": 0,
            "neighbor_count": 0,
            "average_score": math.inf,
        }

    unplaced = [
        component
        for component in components
        if component is not root
    ]
    placed = 0
    placed_pieces = 0
    total_neighbor_count = 0
    total_score = 0.0
    while unplaced:
        best = None
        for component_index, component in enumerate(unplaced):
            component_size = componentSize(component)
            for row_offset, col_offset in iterShiftedComponentPlacements(
                    frame,
                    component,
                    max_shift):
                score, neighbor_count = componentPlacementScore(
                    frame,
                    component,
                    row_offset,
                    col_offset,
                )
                if math.isinf(score):
                    continue
                if neighbor_count < min_neighbor_count:
                    continue
                if max_score is not None and score > max_score:
                    continue
                origin_row, origin_col = componentOrigin(component)
                shift_distance = (
                    abs(row_offset - origin_row)
                    + abs(col_offset - origin_col)
                )
                item = (
                    score,
                    -neighbor_count,
                    shift_distance,
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
            score,
            _neighbor_count,
            _shift_distance,
            _component_size,
            _piece_number,
            row_offset,
            col_offset,
            component_index,
        ) = best
        neighbor_count = -_neighbor_count
        component = unplaced.pop(component_index)
        placeComponent(frame, component, row_offset, col_offset)
        placed += 1
        placed_pieces += componentSize(component)
        total_neighbor_count += neighbor_count
        total_score += score * neighbor_count

    root.pic_connection_matrix = frame
    root.binary_connection_matrix = (frame != 0).astype(int)
    root._kruskal_component_data = None
    return root, unplaced, {
        "placed_components": placed,
        "placed_pieces": placed_pieces,
        "neighbor_count": total_neighbor_count,
        "average_score": averageScore(total_score, total_neighbor_count),
    }


def repairConsensusShifts(
        segment_list,
        top_k=5,
        min_local_support=1,
        min_neighbor_count=3,
        max_shift=1,
        max_score=None):
    if not segment_list:
        return {
            "before": 0,
            "after": 0,
            "cut_edges": 0,
            "split_components": 0,
            "placed_components": 0,
            "placed_pieces": 0,
            "neighbor_count": 0,
            "average_score": math.inf,
        }

    next_component_id = max(segment.component_id for segment in segment_list) + 1
    repaired_segments = []
    cut_edges = 0
    split_components = 0
    placed_components = 0
    placed_pieces = 0
    neighbor_count = 0
    total_score = 0.0

    for segment in segment_list:
        split_segments, segment_cut_edges, next_component_id = (
            splitSegmentByConsensus(
                segment,
                top_k,
                min_local_support,
                next_component_id,
            )
        )
        cut_edges += segment_cut_edges
        if len(split_segments) <= 1:
            repaired_segments.extend(split_segments)
            continue

        split_components += 1
        root, unplaced, placement_stats = placeConsensusComponentsByShift(
            split_segments,
            min_neighbor_count=min_neighbor_count,
            max_shift=max_shift,
            max_score=max_score,
        )
        if root is not None:
            repaired_segments.append(root)
        repaired_segments.extend(unplaced)
        placed_components += placement_stats["placed_components"]
        placed_pieces += placement_stats["placed_pieces"]
        neighbor_count += placement_stats["neighbor_count"]
        if not math.isinf(placement_stats["average_score"]):
            total_score += (
                placement_stats["average_score"]
                * placement_stats["neighbor_count"]
            )

    before = len(segment_list)
    segment_list[:] = repaired_segments
    return {
        "before": before,
        "after": len(segment_list),
        "cut_edges": cut_edges,
        "split_components": split_components,
        "placed_components": placed_components,
        "placed_pieces": placed_pieces,
        "neighbor_count": neighbor_count,
        "average_score": averageScore(total_score, neighbor_count),
    }


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


def stripEmptyBorder(segment):
    frame = segment.pic_connection_matrix
    if not np.any(frame != 0):
        return
    frame = frame[~np.all(frame == 0, axis=1)]
    frame = frame[:, ~np.all(frame == 0, axis=0)]
    segment.pic_connection_matrix = frame
    segment.binary_connection_matrix = (frame != 0).astype(int)


def placeComponentsInExpandedFrame(segment, components):
    if not components:
        return 0, [], []

    frame = segment.pic_connection_matrix
    max_component_height = max(
        component.pic_connection_matrix.shape[0]
        for component in components
    )
    max_component_width = max(
        component.pic_connection_matrix.shape[1]
        for component in components
    )
    height, width = frame.shape
    expanded = np.zeros(
        (
            height + 2 * max_component_height,
            width + 2 * max_component_width,
        ),
        dtype=object,
    )
    expanded[
        max_component_height:max_component_height + height,
        max_component_width:max_component_width + width,
    ] = frame
    (
        placed,
        _placed_pieces,
        unplaced,
        displaced_pieces,
        _neighbor_count,
        _score,
    ) = placeComponentsInFrame(expanded, components)
    segment.pic_connection_matrix = expanded
    stripEmptyBorder(segment)
    return placed, unplaced, displaced_pieces


def averageScore(total_score, count):
    if count == 0:
        return math.inf
    return total_score / count


def placeComponentsInBestFrame(
        segment_list,
        border_tiebreak=False,
        edge_preserving=False):
    best = None
    for root_index, root in enumerate(segment_list):
        components = [
            component
            for index, component in enumerate(segment_list)
            if index != root_index
        ]
        for row_start, col_start in trimFrameStarts(root):
            frame, trimmed = trimFrame(root, row_start, col_start)
            (
                placed,
                placed_pieces,
                unplaced,
                displaced,
                neighbor_count,
                placement_score,
            ) = placeComponentsInFrame(frame, components)
            occupied = int(np.count_nonzero(frame))
            adjacency_count, adjacency_score, score_count = (
                frameAdjacencyScoreStats(frame)
            )
            border_score, border_count = (
                frameBorderScore(frame) if border_tiebreak else (0.0, 0)
            )
            root_preserved_pieces = componentSize(root) - len(trimmed)
            if edge_preserving:
                key = (
                    adjacency_count,
                    score_count,
                    -adjacency_score,
                    neighbor_count,
                    -averageScore(placement_score, neighbor_count),
                    placed,
                    placed_pieces,
                    root_preserved_pieces,
                    -len(trimmed),
                    -len(displaced),
                    occupied,
                    border_score,
                    border_count,
                    -componentSize(root),
                    -abs(row_start),
                    -abs(col_start),
                    -root.piece_number,
                )
            else:
                key = (
                    placed,
                    placed_pieces,
                    root_preserved_pieces,
                    -len(trimmed),
                    -len(displaced),
                    occupied,
                    neighbor_count,
                    -averageScore(placement_score, neighbor_count),
                    adjacency_count,
                    score_count,
                    -adjacency_score,
                    border_score,
                    border_count,
                    -componentSize(root),
                    -abs(row_start),
                    -abs(col_start),
                    -root.piece_number,
                )
            if best is None or key > best[0]:
                best = (
                    key,
                    root,
                    frame,
                    trimmed + displaced,
                    placed,
                    unplaced,
                )

    if best is None:
        return None

    _key, root, frame, trimmed_pieces, placed_components, unplaced = best
    root.pic_connection_matrix = frame
    root.binary_connection_matrix = (frame != 0).astype(int)
    return root, trimmed_pieces, placed_components, unplaced


def trimToBestFrameWithComponents(
        segment,
        components,
        allow_component_overlap=True,
        score_tiebreak=False,
        border_tiebreak=False,
        edge_preserving=False):
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
            allow_overlap=allow_component_overlap,
        )
        occupied = int(np.count_nonzero(frame))
        if score_tiebreak or border_tiebreak:
            frame_neighbor_count, frame_score, frame_score_count = (
                frameAdjacencyScoreStats(frame)
            )
            frame_border_score, frame_border_count = (
                frameBorderScore(frame) if border_tiebreak else (0.0, 0)
            )
        else:
            frame_neighbor_count = 0
            frame_score = 0.0
            frame_score_count = 0
            frame_border_score = 0.0
            frame_border_count = 0
        if edge_preserving:
            key = (
                frame_neighbor_count,
                frame_score_count,
                -frame_score,
                neighbor_count,
                -score,
                occupied,
                placed_pieces,
                placed,
                root_occupied,
                frame_border_score,
                frame_border_count,
                -len(trimmed),
                -abs(row_start),
                -abs(col_start),
            )
        else:
            key = (
                occupied,
                placed_pieces,
                placed,
                frame_border_score,
                frame_border_count,
                frame_neighbor_count,
                frame_score_count,
                -frame_score,
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
        preserve_components=False,
        conservative_fill=False,
        border_tiebreak=False,
        component_frame_search=False,
        edge_preserving=False):
    if not segment_list:
        return None

    original_snapshot = None
    original_stats = None
    if edge_preserving:
        original_snapshot = snapshotSegments(segment_list)
        original_stats = segmentListAdjacencyScoreStats(segment_list)

    placed_components = 0
    trimmed_pieces = []
    if component_frame_search and len(segment_list) > 1:
        component_frame = placeComponentsInBestFrame(
            segment_list,
            border_tiebreak=border_tiebreak,
            edge_preserving=edge_preserving,
        )
        if component_frame is not None:
            (
                root,
                trimmed_pieces,
                placed_components,
                leftover_components,
            ) = component_frame
        else:
            root = max(segment_list, key=componentSize)
            leftover_components = [
                segment
                for segment in segment_list
                if segment is not root
            ]
    else:
        root = max(segment_list, key=componentSize)
        leftover_components = []
        for segment in segment_list:
            if segment is root:
                continue
            leftover_components.append(segment)
        if preserve_components and conservative_fill and leftover_components:
            pre_placed, leftover_components, displaced_pieces = (
                placeComponentsInExpandedFrame(root, leftover_components)
            )
            placed_components += pre_placed
            trimmed_pieces.extend(displaced_pieces)
        if preserve_components:
            frame_trimmed_pieces, frame_placed, leftover_components = (
                trimToBestFrameWithComponents(
                    root,
                    leftover_components,
                    allow_component_overlap=not conservative_fill,
                    score_tiebreak=conservative_fill,
                    border_tiebreak=border_tiebreak,
                    edge_preserving=edge_preserving,
                )
            )
            trimmed_pieces.extend(frame_trimmed_pieces)
            placed_components += frame_placed
        else:
            trimmed_pieces = trimToBestFrame(
                root,
                score_tiebreak=conservative_fill,
                border_tiebreak=border_tiebreak,
                edge_preserving=edge_preserving,
            )
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
    fill_score_limit = None
    if conservative_fill or edge_preserving:
        fill_score_limit = conservativeFillScoreLimit(root.pic_connection_matrix)
        if show_progress and fill_score_limit is not None:
            print(
                "Trim/fill post-process: "
                f"conservative fill score limit {fill_score_limit:.6g}",
                flush=True,
            )
    min_neighbor_count = 2 if edge_preserving else 1
    filled = fillHoles(
        root,
        candidates,
        show_progress=show_progress,
        max_score=fill_score_limit,
        min_neighbor_count=min_neighbor_count,
    )

    if edge_preserving:
        final_stats = segmentListAdjacencyScoreStats([root])
        if shouldKeepOriginalAssembly(original_stats, final_stats):
            restoreSegments(segment_list, original_snapshot)
            if show_progress:
                print(
                    "Trim/fill post-process: "
                    "kept original assembly because trim/fill dropped too "
                    "many retained edges",
                    flush=True,
                )
            if len(segment_list) == 1:
                original_root = segment_list[0]
                return BestConnection(
                    own_segment=original_root,
                    pic_connection_matrix=original_root.pic_connection_matrix,
                    binary_connection_matrix=original_root.binary_connection_matrix,
                )
            return None

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
