import heapq
import itertools
import random

import numpy as np

from .enums import CompareWithOtherSegments, JoinDirection, OPPOSITE_DIRECTIONS
from .models import BestConnection, Segment, scoreValuesArray
from .score_table import DIRECTIONS_BY_INDEX, DIRECTION_TO_INDEX


DIRECTION_DELTAS = {
    JoinDirection.UP: (-1, 0),
    JoinDirection.DOWN: (1, 0),
    JoinDirection.LEFT: (0, -1),
    JoinDirection.RIGHT: (0, 1),
}


def _segmentPieceNumbers(segment):
    return {
        int(piece_number)
        for piece_number
        in segment.kruskalComponentData()["piece_by_position"].values()
    }


def _segmentExposedEdges(segment):
    data = segment.kruskalComponentData()
    positions = set(data["positions"])
    piece_by_position = data["piece_by_position"]
    for row, col in data["boundary_piece_positions"]:
        piece_number = int(piece_by_position[(row, col)])
        for direction, (row_delta, col_delta) in DIRECTION_DELTAS.items():
            if (row + row_delta, col + col_delta) not in positions:
                yield piece_number, direction


def _scoreValue(score_dict, score_values, own_number, direction, join_number):
    if score_values is not None:
        return float(
            score_values[
                own_number,
                DIRECTIONS_BY_INDEX.index(direction),
                join_number,
            ]
        )
    return float(score_dict[own_number, direction, join_number])


def _bestOutgoingScore(
        score_dict,
        score_values,
        own_number,
        direction,
        candidate_piece_numbers):
    best_score = float("inf")
    for join_number in candidate_piece_numbers:
        try:
            score = _scoreValue(
                score_dict,
                score_values,
                own_number,
                direction,
                join_number,
            )
        except KeyError:
            continue
        if np.isnan(score):
            continue
        if score < best_score:
            best_score = score
    return best_score


def primNeighborhoodSeedScore(
        segment,
        segment_list,
        neighborhood_size=4,
        own_piece_numbers=None,
        candidate_piece_numbers=None):
    score_dict = segment.score_dict
    score_values = scoreValuesArray(score_dict)
    if own_piece_numbers is None:
        own_piece_numbers = _segmentPieceNumbers(segment)
    if candidate_piece_numbers is None:
        candidate_piece_numbers = [
            candidate_piece
            for compare_segment in segment_list
            if compare_segment != segment
            for candidate_piece in _segmentPieceNumbers(compare_segment)
            if candidate_piece not in own_piece_numbers
        ]
    if not candidate_piece_numbers:
        return float("inf")

    outgoing_scores = []
    for own_number, direction in _segmentExposedEdges(segment):
        score = _bestOutgoingScore(
            score_dict,
            score_values,
            own_number,
            direction,
            candidate_piece_numbers,
        )
        if not np.isinf(score):
            outgoing_scores.append(score)
    if not outgoing_scores:
        return float("inf")

    outgoing_scores.sort()
    neighborhood_size = max(1, min(neighborhood_size, len(outgoing_scores)))
    return float(np.mean(outgoing_scores[:neighborhood_size]))


def findBestPrimSeedSegment(
        segment_list,
        strategy="neighborhood",
        neighborhood_size=4):
    if not segment_list:
        return None
    if strategy == "random":
        return random.choice(segment_list)
    if strategy != "neighborhood":
        raise ValueError("Unknown Prim seed strategy")
    piece_numbers_by_segment = {
        segment: _segmentPieceNumbers(segment)
        for segment in segment_list
    }
    all_piece_numbers = set()
    for piece_numbers in piece_numbers_by_segment.values():
        all_piece_numbers.update(piece_numbers)

    def seed_key(segment):
        own_piece_numbers = piece_numbers_by_segment[segment]
        candidate_piece_numbers = [
            piece_number
            for piece_number in all_piece_numbers
            if piece_number not in own_piece_numbers
        ]
        return (
            primNeighborhoodSeedScore(
                segment,
                segment_list,
                neighborhood_size=neighborhood_size,
                own_piece_numbers=own_piece_numbers,
                candidate_piece_numbers=candidate_piece_numbers,
            ),
            -len(own_piece_numbers),
            segment.piece_number,
        )

    return min(segment_list, key=seed_key)


def primConnectionEstimate(root_segment, join_segment):
    score_dict = root_segment.score_dict
    score_values = scoreValuesArray(score_dict)
    join_number = join_segment.piece_number
    best_score = float("inf")
    for own_number, direction in _segmentExposedEdges(root_segment):
        for score_owner, score_direction, score_join in (
                (own_number, direction, join_number),
                (join_number, OPPOSITE_DIRECTIONS[direction], own_number)):
            try:
                score = _scoreValue(
                    score_dict,
                    score_values,
                    score_owner,
                    score_direction,
                    score_join,
                )
            except KeyError:
                continue
            if np.isnan(score):
                continue
            if score < best_score:
                best_score = score
    return best_score


def primConnectionEstimates(root_segment, join_segments):
    if not join_segments:
        return []
    score_dict = root_segment.score_dict
    score_values = scoreValuesArray(score_dict)
    if score_values is None:
        return [
            primConnectionEstimate(root_segment, join_segment)
            for join_segment in join_segments
        ]

    exposed_edges = tuple(_segmentExposedEdges(root_segment))
    if not exposed_edges:
        return [float("inf")] * len(join_segments)
    own_numbers = np.asarray(
        [own_number for own_number, _direction in exposed_edges],
        dtype=np.intp,
    )
    direction_indices = np.asarray(
        [DIRECTION_TO_INDEX[direction] for _own_number, direction in exposed_edges],
        dtype=np.intp,
    )
    opposite_indices = np.asarray(
        [
            DIRECTION_TO_INDEX[OPPOSITE_DIRECTIONS[direction]]
            for _own_number, direction in exposed_edges
        ],
        dtype=np.intp,
    )
    join_numbers = np.asarray(
        [join_segment.piece_number for join_segment in join_segments],
        dtype=np.intp,
    )
    forward = score_values[
        own_numbers[:, None],
        direction_indices[:, None],
        join_numbers[None, :],
    ]
    reverse = score_values[
        join_numbers[None, :],
        opposite_indices[:, None],
        own_numbers[:, None],
    ]
    combined = np.minimum(forward, reverse)
    estimates = np.nanmin(combined, axis=0)
    return [float(estimate) for estimate in estimates]


class KruskalBeamState:
    def __init__(
            self,
            segment_list,
            connection_queue=None,
            path_score=0.0,
            rounds=0,
            history=()):
        self.segment_list = segment_list
        self.connection_queue = connection_queue
        self.path_score = path_score
        self.rounds = rounds
        self.history = history


def connectionPriority(connection, compare_type):
    if compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
        return (
            2 * (connection.score * (
                connection.score / connection.second_best_score
            ))
        ) + connection.score
    return connection.score


def findBestConnectionKruskal(segment_list, compare_type, boost_priority_of_big_pieces_joining, compare_mode):
    best_so_far = BestConnection()
    if compare_mode == CompareWithOtherSegments.ONLY_BEST:
        for index, segment1 in enumerate(segment_list):
            for segment2 in segment_list[index+1:]:
                segment1.best_connection_found_so_far = BestConnection()
                temp = segment1.calculateConnectionsKruskal(
                    segment2, boost_priority_of_big_pieces_joining)
                if temp.isBetterConnection(best_so_far, compare_type):
                    best_so_far = temp
        return best_so_far
    else:
        for segment1 in segment_list:
            for segment2 in segment_list:
                if segment1 != segment2:
                    segment1.best_connection_found_so_far = BestConnection()
                    temp = segment1.calculateConnectionsKruskal(
                        segment2, boost_priority_of_big_pieces_joining)
                    if temp.isBetterConnection(best_so_far, compare_type):
                        best_so_far = temp
        return best_so_far


class KruskalConnectionPriorityQueue:
    def __init__(
            self,
            segment_list,
            boost_priority_of_big_pieces_joining=False,
            compare_type=CompareWithOtherSegments.ONLY_BEST,
            compare_mode=CompareWithOtherSegments.ONLY_BEST,
            min_contact_count=1,
            max_score=None):
        self.boost_priority_of_big_pieces_joining = boost_priority_of_big_pieces_joining
        self.compare_type = compare_type
        self.compare_mode = compare_mode
        self.min_contact_count = min_contact_count
        self.max_score = max_score
        self._counter = itertools.count()
        self._heap = []
        self.addAllConnections(segment_list)

    def addAllConnections(self, segment_list):
        if self.compare_mode == CompareWithOtherSegments.ONLY_BEST:
            for index, segment1 in enumerate(segment_list):
                for segment2 in segment_list[index+1:]:
                    self._pushConnection(segment1, segment2)
        else:
            for segment1 in segment_list:
                for segment2 in segment_list:
                    if segment1 != segment2:
                        self._pushConnection(segment1, segment2)

    def addConnectionsFor(self, updated_segment, segment_list):
        if self.compare_mode != CompareWithOtherSegments.ONLY_BEST:
            for segment in segment_list:
                if segment is updated_segment:
                    continue
                self._pushConnection(updated_segment, segment)
                self._pushConnection(segment, updated_segment)
            return

        updated_index = segment_list.index(updated_segment)
        for index, segment in enumerate(segment_list):
            if segment is updated_segment:
                continue
            if updated_index < index:
                self._pushConnection(updated_segment, segment)
            else:
                self._pushConnection(segment, updated_segment)

    def popBestConnection(self, segment_list):
        active_segments = set(segment_list)
        while self._heap:
            item = heapq.heappop(self._heap)
            connection = self._connectionFromItem(
                item,
                active_segments,
                materialize=True,
            )
            if connection is not None:
                return connection
        return BestConnection()

    def topConnections(self, segment_list, limit):
        if limit <= 0:
            return []

        active_segments = set(segment_list)
        connections = []
        retained_items = []
        while self._heap and len(connections) < limit:
            item = heapq.heappop(self._heap)
            connection = self._connectionFromItem(
                item,
                active_segments,
                materialize=False,
            )
            if connection is not None:
                retained_items.append(item)
                connections.append(connection)
        for item in retained_items:
            heapq.heappush(self._heap, item)
        return connections

    def cloneForBranch(self, segment_mapping, excluded_segments):
        cloned = self.__class__.__new__(self.__class__)
        cloned.boost_priority_of_big_pieces_joining = (
            self.boost_priority_of_big_pieces_joining
        )
        cloned.compare_type = self.compare_type
        cloned.compare_mode = self.compare_mode
        cloned.min_contact_count = self.min_contact_count
        cloned.max_score = self.max_score
        cloned_heap = []
        counter = 0
        append_heap = cloned_heap.append
        segment_mapping_get = segment_mapping.get

        excluded_segments = set(excluded_segments)
        for (
                priority,
                _counter,
                _connection,
                own_segment,
                join_segment,
                own_component_id,
                join_component_id,
        ) in self._heap:
            if own_segment in excluded_segments or join_segment in excluded_segments:
                continue
            if own_segment.component_id != own_component_id:
                continue
            if join_segment.component_id != join_component_id:
                continue

            cloned_own_segment = segment_mapping_get(own_segment)
            cloned_join_segment = segment_mapping_get(join_segment)
            if cloned_own_segment is None or cloned_join_segment is None:
                continue
            append_heap(
                (
                    priority,
                    counter,
                    None,
                    cloned_own_segment,
                    cloned_join_segment,
                    cloned_own_segment.component_id,
                    cloned_join_segment.component_id,
                )
            )
            counter += 1
        heapq.heapify(cloned_heap)
        cloned._counter = itertools.count(counter)
        cloned._heap = cloned_heap
        return cloned

    def _connectionFromItem(self, item, active_segments, materialize):
        (
            _priority,
            _counter,
            connection,
            own_segment,
            join_segment,
            own_component_id,
            join_component_id,
        ) = item
        if own_segment not in active_segments or join_segment not in active_segments:
            return None
        if own_segment.component_id != own_component_id:
            return None
        if join_segment.component_id != join_component_id:
            return None
        if connection is None:
            own_segment.best_connection_found_so_far = BestConnection()
            connection = own_segment.calculateConnectionsKruskal(
                join_segment,
                self.boost_priority_of_big_pieces_joining,
                defer_connection_matrices=not materialize,
            )
            if not connection.hasConnection():
                return None
        elif materialize:
            own_segment.materializeKruskalConnection(connection)
        return connection

    def _pushConnection(self, segment1, segment2):
        segment1.best_connection_found_so_far = BestConnection()
        connection = segment1.calculateConnectionsKruskal(
            segment2,
            self.boost_priority_of_big_pieces_joining,
            defer_connection_matrices=True,
        )
        if not connection.hasConnection():
            return
        if connection.contact_count < self.min_contact_count:
            return
        if self.max_score is not None and connection.score > self.max_score:
            return
        heapq.heappush(
            self._heap,
            (
                self._connectionPriority(connection),
                next(self._counter),
                connection,
                connection.own_segment,
                connection.join_segment,
                connection.own_segment.component_id,
                connection.join_segment.component_id,
            ),
        )

    def _connectionPriority(self, connection):
        return connectionPriority(connection, self.compare_type)


class PrimConnectionPriorityQueue:
    ESTIMATE = 0
    EXACT = 1

    def __init__(
            self,
            root_segment,
            segment_list,
            compare_type=CompareWithOtherSegments.ONLY_BEST):
        self.compare_type = compare_type
        self._counter = itertools.count()
        self._heap = []
        self.addConnectionsForRoot(root_segment, segment_list)

    def addConnectionsForRoot(self, root_segment, segment_list):
        join_segments = [
            segment
            for segment in segment_list
            if segment is not root_segment
        ]
        if self.compare_type == CompareWithOtherSegments.ONLY_BEST:
            estimates = primConnectionEstimates(root_segment, join_segments)
            for join_segment, estimate in zip(join_segments, estimates):
                self._pushEstimatedConnection(
                    root_segment,
                    join_segment,
                    estimate,
                )
            return

        for join_segment in join_segments:
            self._pushExactConnectionFor(root_segment, join_segment)

    def popBestConnection(self, root_segment, segment_list):
        active_segments = set(segment_list)
        while self._heap:
            item = heapq.heappop(self._heap)
            connection = self._connectionFromItem(
                item,
                root_segment,
                active_segments,
            )
            if connection is not None:
                return connection
        return BestConnection()

    def _connectionPriority(self, connection):
        return connectionPriority(connection, self.compare_type)

    def _connectionFromItem(self, item, root_segment, active_segments):
        (
            _priority,
            _counter,
            item_type,
            connection,
            queued_root_segment,
            join_segment,
            root_component_id,
            join_component_id,
        ) = item
        if queued_root_segment is not root_segment:
            return None
        if queued_root_segment not in active_segments:
            return None
        if join_segment not in active_segments:
            return None
        if queued_root_segment.component_id != root_component_id:
            return None
        if join_segment.component_id != join_component_id:
            return None
        if item_type == self.EXACT:
            return connection

        queued_root_segment.best_connection_found_so_far = BestConnection()
        connection = queued_root_segment.calculateConnectionsPrim(join_segment)
        if connection.pic_connection_matrix is None:
            return None
        self._pushExactConnection(connection)
        return None

    def _pushConnection(self, root_segment, join_segment):
        if self.compare_type == CompareWithOtherSegments.ONLY_BEST:
            self._pushEstimatedConnection(
                root_segment,
                join_segment,
                primConnectionEstimate(root_segment, join_segment),
            )
        else:
            self._pushExactConnectionFor(root_segment, join_segment)

    def _pushEstimatedConnection(self, root_segment, join_segment, priority):
        if np.isinf(priority):
            return
        heapq.heappush(
            self._heap,
            (
                priority,
                next(self._counter),
                self.ESTIMATE,
                None,
                root_segment,
                join_segment,
                root_segment.component_id,
                join_segment.component_id,
            ),
        )

    def _pushExactConnectionFor(self, root_segment, join_segment):
        root_segment.best_connection_found_so_far = BestConnection()
        connection = root_segment.calculateConnectionsPrim(join_segment)
        if connection.pic_connection_matrix is None:
            return
        self._pushExactConnection(connection)

    def _pushExactConnection(self, connection):
        heapq.heappush(
            self._heap,
            (
                self._connectionPriority(connection),
                next(self._counter),
                self.EXACT,
                connection,
                connection.own_segment,
                connection.join_segment,
                connection.own_segment.component_id,
                connection.join_segment.component_id,
            ),
        )


def assembleKruskalWithPriorityQueue(
        segment_list,
        original_size,
        boost_priority_of_big_pieces_joining=False,
        compare_type=CompareWithOtherSegments.ONLY_BEST,
        compare_mode=CompareWithOtherSegments.ONLY_BEST,
        on_join=None):
    connection_queue = KruskalConnectionPriorityQueue(
        segment_list,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
    )
    rounds = 0
    while len(segment_list) > 1:
        best_connection = connection_queue.popBestConnection(segment_list)
        if best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        connection_queue.addConnectionsFor(
            best_connection.own_segment,
            segment_list,
        )
        if on_join is not None:
            on_join(best_connection, rounds)
        rounds += 1
    return rounds


def assembleKruskalStaged(
        segment_list,
        original_size,
        boost_priority_of_big_pieces_joining=False,
        compare_type=CompareWithOtherSegments.ONLY_BEST,
        compare_mode=CompareWithOtherSegments.ONLY_BEST,
        min_contact_count=2,
        max_multi_contact_score=None,
        on_join=None):
    multi_contact_queue = KruskalConnectionPriorityQueue(
        segment_list,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
        min_contact_count=min_contact_count,
        max_score=max_multi_contact_score,
    )
    fallback_queue = KruskalConnectionPriorityQueue(
        segment_list,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
    )
    rounds = 0
    while len(segment_list) > 1:
        best_connection = multi_contact_queue.popBestConnection(segment_list)
        if best_connection.pic_connection_matrix is None:
            best_connection = fallback_queue.popBestConnection(segment_list)
        if best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        multi_contact_queue.addConnectionsFor(
            best_connection.own_segment,
            segment_list,
        )
        fallback_queue.addConnectionsFor(
            best_connection.own_segment,
            segment_list,
        )
        if on_join is not None:
            on_join(best_connection, rounds)
        rounds += 1
    return rounds


def assembleKruskalMultiContactOnly(
        segment_list,
        original_size,
        boost_priority_of_big_pieces_joining=False,
        compare_type=CompareWithOtherSegments.ONLY_BEST,
        compare_mode=CompareWithOtherSegments.ONLY_BEST,
        min_contact_count=2,
        max_score=None,
        on_join=None):
    connection_queue = KruskalConnectionPriorityQueue(
        segment_list,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
        min_contact_count=min_contact_count,
        max_score=max_score,
    )
    rounds = 0
    while len(segment_list) > 1:
        best_connection = connection_queue.popBestConnection(segment_list)
        if best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        connection_queue.addConnectionsFor(
            best_connection.own_segment,
            segment_list,
        )
        if on_join is not None:
            on_join(best_connection, rounds)
        rounds += 1
    return rounds


def iterComponentPieces(segment):
    for row, col in zip(*np.where(segment.pic_connection_matrix != 0)):
        yield int(row), int(col), segment.pic_connection_matrix[row, col]


def piecePositionsInSegment(segment):
    return {
        piece.piece_number: (row, col)
        for row, col, piece in iterComponentPieces(segment)
    }


def updatePieceSegmentMap(piece_to_segment, segment):
    for _row, _col, piece in iterComponentPieces(segment):
        piece_to_segment[piece.piece_number] = segment


def pairwiseKruskalCandidateEdges(
        segment_list,
        top_candidates_per_edge=10,
        mutual_edges_only=False,
        max_score=None):
    if not segment_list:
        return []
    score_dict = segment_list[0].score_dict
    score_values = scoreValuesArray(score_dict)
    piece_count = segment_list[0].max_width * segment_list[0].max_height
    candidate_limit = min(top_candidates_per_edge, piece_count - 1)
    edges = []
    counter = itertools.count()

    if score_values is not None:
        best_by_piece_direction = None
        if mutual_edges_only:
            best_by_piece_direction = np.full(
                (piece_count + 1, len(DIRECTIONS_BY_INDEX)),
                -1,
                dtype=np.int32,
            )
            for own_piece in range(1, piece_count + 1):
                for direction_index in range(len(DIRECTIONS_BY_INDEX)):
                    scores = np.asarray(
                        score_values[own_piece, direction_index, :],
                        dtype=np.float64,
                    ).copy()
                    scores[0] = np.inf
                    scores[own_piece] = np.inf
                    scores[np.isnan(scores)] = np.inf
                    best_by_piece_direction[own_piece, direction_index] = (
                        int(np.argmin(scores))
                    )
        for own_piece in range(1, piece_count + 1):
            for direction_index, direction in enumerate(DIRECTIONS_BY_INDEX):
                scores = np.asarray(
                    score_values[own_piece, direction_index, :],
                    dtype=np.float64,
                ).copy()
                scores[0] = np.inf
                scores[own_piece] = np.inf
                scores[np.isnan(scores)] = np.inf
                if candidate_limit <= 0:
                    continue
                candidate_indices = np.argpartition(
                    scores,
                    candidate_limit - 1,
                )[:candidate_limit]
                for join_piece in candidate_indices[
                        np.argsort(scores[candidate_indices])]:
                    score = float(scores[join_piece])
                    if np.isinf(score):
                        continue
                    if max_score is not None and score > max_score:
                        continue
                    if mutual_edges_only:
                        opposite_index = DIRECTIONS_BY_INDEX.index(
                            OPPOSITE_DIRECTIONS[direction],
                        )
                        if (
                                best_by_piece_direction[
                                    int(join_piece),
                                    opposite_index,
                                ]
                                != own_piece):
                            continue
                    edges.append((
                        score,
                        next(counter),
                        own_piece,
                        direction,
                        int(join_piece),
                    ))
    else:
        for own_piece in range(1, piece_count + 1):
            for direction in JoinDirection:
                candidates = []
                for join_piece in range(1, piece_count + 1):
                    if join_piece == own_piece:
                        continue
                    score = score_dict.get(
                        (own_piece, direction, join_piece),
                        None,
                    )
                    if score is None:
                        continue
                    candidates.append((score, join_piece))
                for score, join_piece in heapq.nsmallest(
                        candidate_limit,
                        candidates):
                    if max_score is not None and score > max_score:
                        continue
                    edges.append((
                        float(score),
                        next(counter),
                        own_piece,
                        direction,
                        join_piece,
                    ))

    heapq.heapify(edges)
    return edges


def _finiteScore(score_values, own_piece, direction, join_piece):
    direction_index = DIRECTION_TO_INDEX[direction]
    score = float(score_values[own_piece, direction_index, join_piece])
    if np.isnan(score) or np.isinf(score):
        return None
    return score


def _topCandidateMap(segment_list, top_candidates_per_edge=10, max_score=None):
    if not segment_list:
        return {}, {}, 0
    score_dict = segment_list[0].score_dict
    score_values = scoreValuesArray(score_dict)
    piece_count = segment_list[0].max_width * segment_list[0].max_height
    candidate_limit = min(top_candidates_per_edge, piece_count - 1)
    candidates = {}
    scores = {}

    if score_values is not None:
        for own_piece in range(1, piece_count + 1):
            for direction_index, direction in enumerate(DIRECTIONS_BY_INDEX):
                direction_scores = np.asarray(
                    score_values[own_piece, direction_index, :],
                    dtype=np.float64,
                ).copy()
                direction_scores[0] = np.inf
                direction_scores[own_piece] = np.inf
                direction_scores[np.isnan(direction_scores)] = np.inf
                finite = np.isfinite(direction_scores)
                if not np.any(finite) or candidate_limit <= 0:
                    candidates[own_piece, direction] = ()
                    continue
                limit = min(candidate_limit, int(np.count_nonzero(finite)))
                candidate_indices = np.argpartition(
                    direction_scores,
                    limit - 1,
                )[:limit]
                ordered = []
                for join_piece in candidate_indices[
                        np.argsort(direction_scores[candidate_indices])]:
                    score = float(direction_scores[join_piece])
                    if max_score is not None and score > max_score:
                        continue
                    join_piece = int(join_piece)
                    ordered.append(join_piece)
                    scores[own_piece, direction, join_piece] = score
                candidates[own_piece, direction] = tuple(ordered)
        return candidates, scores, piece_count

    for own_piece in range(1, piece_count + 1):
        for direction in JoinDirection:
            direction_candidates = []
            for join_piece in range(1, piece_count + 1):
                if join_piece == own_piece:
                    continue
                score = score_dict.get((own_piece, direction, join_piece), None)
                if score is None:
                    continue
                score = float(score[0] if isinstance(score, tuple) else score)
                if np.isnan(score) or np.isinf(score):
                    continue
                if max_score is not None and score > max_score:
                    continue
                direction_candidates.append((score, join_piece))
            ordered = []
            for score, join_piece in heapq.nsmallest(
                    candidate_limit,
                    direction_candidates):
                ordered.append(join_piece)
                scores[own_piece, direction, join_piece] = score
            candidates[own_piece, direction] = tuple(ordered)
    return candidates, scores, piece_count


def _scoreForConsensusEdge(segment_list, scores, own_piece, direction, join_piece):
    score = scores.get((own_piece, direction, join_piece))
    if score is not None:
        return score
    score_dict = segment_list[0].score_dict
    score_values = scoreValuesArray(score_dict)
    if score_values is not None:
        score = _finiteScore(score_values, own_piece, direction, join_piece)
        if score is not None:
            scores[own_piece, direction, join_piece] = score
        return score
    score = score_dict.get((own_piece, direction, join_piece), None)
    if score is None:
        return None
    score = float(score[0] if isinstance(score, tuple) else score)
    if np.isnan(score) or np.isinf(score):
        return None
    scores[own_piece, direction, join_piece] = score
    return score


def growingConsensusCandidateEdges(
        segment_list,
        top_candidates_per_edge=10,
        min_support=1,
        propose_missing=True,
        max_score=None,
        max_edges=None,
        priority="score"):
    candidates, scores, piece_count = _topCandidateMap(
        segment_list,
        top_candidates_per_edge=top_candidates_per_edge,
        max_score=max_score,
    )
    if piece_count == 0:
        return []

    support = {}

    def add_supported_edge(own_piece, direction, join_piece):
        if own_piece == join_piece:
            return False
        score = _scoreForConsensusEdge(
            segment_list,
            scores,
            own_piece,
            direction,
            join_piece,
        )
        if score is None:
            return False
        if max_score is not None and score > max_score:
            return False
        support[own_piece, direction, join_piece] = (
            support.get((own_piece, direction, join_piece), 0) + 1
        )
        opposite = OPPOSITE_DIRECTIONS[direction]
        reverse_score = _scoreForConsensusEdge(
            segment_list,
            scores,
            join_piece,
            opposite,
            own_piece,
        )
        if reverse_score is not None and (
                max_score is None or reverse_score <= max_score):
            support[join_piece, opposite, own_piece] = (
                support.get((join_piece, opposite, own_piece), 0) + 1
            )
        return True

    for top_left in range(1, piece_count + 1):
        right_pieces = candidates.get((top_left, JoinDirection.RIGHT), ())
        down_pieces = candidates.get((top_left, JoinDirection.DOWN), ())
        for top_right in right_pieces:
            for bottom_left in down_pieces:
                if bottom_left == top_right:
                    continue
                bottom_rights = set(
                    candidates.get((top_right, JoinDirection.DOWN), ())
                )
                if propose_missing:
                    bottom_rights.update(
                        candidates.get((bottom_left, JoinDirection.RIGHT), ())
                    )
                else:
                    bottom_rights.intersection_update(
                        candidates.get((bottom_left, JoinDirection.RIGHT), ())
                    )
                for bottom_right in bottom_rights:
                    if bottom_right in (top_left, top_right, bottom_left):
                        continue
                    has_top_right_down = (
                        bottom_right
                        in candidates.get((top_right, JoinDirection.DOWN), ())
                    )
                    has_bottom_left_right = (
                        bottom_right
                        in candidates.get((bottom_left, JoinDirection.RIGHT), ())
                    )
                    if not (
                            has_top_right_down
                            or has_bottom_left_right):
                        continue
                    if not propose_missing and not (
                            has_top_right_down
                            and has_bottom_left_right):
                        continue

                    add_supported_edge(
                        top_left,
                        JoinDirection.RIGHT,
                        top_right,
                    )
                    add_supported_edge(
                        top_left,
                        JoinDirection.DOWN,
                        bottom_left,
                    )
                    if has_top_right_down or propose_missing:
                        add_supported_edge(
                            top_right,
                            JoinDirection.DOWN,
                            bottom_right,
                        )
                    if has_bottom_left_right or propose_missing:
                        add_supported_edge(
                            bottom_left,
                            JoinDirection.RIGHT,
                            bottom_right,
                        )

    edges = []
    counter = itertools.count()
    for (own_piece, direction, join_piece), edge_support in support.items():
        if edge_support < min_support:
            continue
        score = _scoreForConsensusEdge(
            segment_list,
            scores,
            own_piece,
            direction,
            join_piece,
        )
        if score is None:
            continue
        if max_score is not None and score > max_score:
            continue
        if priority == "support":
            primary = -edge_support
            secondary = score
        elif priority == "score":
            primary = score
            secondary = -edge_support
        else:
            raise ValueError("priority must be 'score' or 'support'")
        edges.append((
            primary,
            secondary,
            next(counter),
            edge_support,
            score,
            own_piece,
            direction,
            join_piece,
        ))
    if max_edges is not None and len(edges) > max_edges:
        edges = heapq.nsmallest(max_edges, edges)
    heapq.heapify(edges)
    return edges


def calculatePairwiseKruskalConnection(
        own_segment,
        join_segment,
        own_piece_number,
        direction,
        join_piece_number,
        score):
    own_positions = piecePositionsInSegment(own_segment)
    join_positions = piecePositionsInSegment(join_segment)
    own_position = own_positions.get(own_piece_number)
    join_position = join_positions.get(join_piece_number)
    if own_position is None or join_position is None:
        return BestConnection()

    row_delta, col_delta = DIRECTION_DELTAS[direction]
    target_join_row = own_position[0] + row_delta
    target_join_col = own_position[1] + col_delta
    join_row_offset = target_join_row - join_position[0]
    join_col_offset = target_join_col - join_position[1]

    own_piece_positions = tuple(iterComponentPieces(own_segment))
    join_piece_positions = tuple(iterComponentPieces(join_segment))
    rows = [row for row, _col, _piece in own_piece_positions]
    cols = [col for _row, col, _piece in own_piece_positions]
    rows.extend(row + join_row_offset for row, _col, _piece in join_piece_positions)
    cols.extend(col + join_col_offset for _row, col, _piece in join_piece_positions)
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if (
            getattr(own_segment, "enforce_frame_bounds", True)
            and (
                height > own_segment.max_height
                or width > own_segment.max_width
            )):
        return BestConnection()

    combined_pointer = np.zeros((height, width), dtype=object)
    combined_binary = np.zeros((height, width))
    for row, col, piece in own_piece_positions:
        combined_row = row - min_row
        combined_col = col - min_col
        combined_pointer[combined_row, combined_col] = piece
        combined_binary[combined_row, combined_col] = 1
    for row, col, piece in join_piece_positions:
        combined_row = row + join_row_offset - min_row
        combined_col = col + join_col_offset - min_col
        if combined_pointer[combined_row, combined_col] != 0:
            return BestConnection()
        combined_pointer[combined_row, combined_col] = piece
        combined_binary[combined_row, combined_col] = 1

    connection = BestConnection(
        own_segment=own_segment,
        pic_connection_matrix=combined_pointer,
        join_segment=join_segment,
        binary_connection_matrix=combined_binary,
    )
    connection.score = score
    connection.second_best_score = score
    connection.contact_count = 1
    return connection


def assembleGallagherPairwiseKruskal(
        segment_list,
        original_size,
        top_candidates_per_edge=10,
        mutual_edges_only=False,
        max_score=None,
        on_join=None):
    candidate_edges = pairwiseKruskalCandidateEdges(
        segment_list,
        top_candidates_per_edge,
        mutual_edges_only,
        max_score,
    )
    piece_to_segment = {}
    for segment in segment_list:
        updatePieceSegmentMap(piece_to_segment, segment)

    rounds = 0
    while len(segment_list) > 1 and candidate_edges:
        score, _counter, own_piece, direction, join_piece = heapq.heappop(
            candidate_edges,
        )
        own_segment = piece_to_segment.get(own_piece)
        join_segment = piece_to_segment.get(join_piece)
        if own_segment is None or join_segment is None:
            continue
        if own_segment is join_segment:
            continue
        connection = calculatePairwiseKruskalConnection(
            own_segment,
            join_segment,
            own_piece,
            direction,
            join_piece,
            score,
        )
        if connection.pic_connection_matrix is None:
            continue
        joinPieces(connection, segment_list, original_size)
        updatePieceSegmentMap(piece_to_segment, connection.own_segment)
        if on_join is not None:
            on_join(connection, rounds)
        rounds += 1
    return rounds


def assembleGrowingConsensusKruskal(
        segment_list,
        original_size,
        top_candidates_per_edge=10,
        min_support=1,
        propose_missing=True,
        max_score=None,
        max_edges=None,
        priority="score",
        fallback_pairwise=False,
        on_join=None):
    candidate_edges = growingConsensusCandidateEdges(
        segment_list,
        top_candidates_per_edge=top_candidates_per_edge,
        min_support=min_support,
        propose_missing=propose_missing,
        max_score=max_score,
        max_edges=max_edges,
        priority=priority,
    )
    piece_to_segment = {}
    for segment in segment_list:
        updatePieceSegmentMap(piece_to_segment, segment)

    rounds = 0
    while len(segment_list) > 1 and candidate_edges:
        (
            _primary,
            _secondary,
            _counter,
            edge_support,
            score,
            own_piece,
            direction,
            join_piece,
        ) = (
            heapq.heappop(candidate_edges)
        )
        own_segment = piece_to_segment.get(own_piece)
        join_segment = piece_to_segment.get(join_piece)
        if own_segment is None or join_segment is None:
            continue
        if own_segment is join_segment:
            continue
        connection = calculatePairwiseKruskalConnection(
            own_segment,
            join_segment,
            own_piece,
            direction,
            join_piece,
            score,
        )
        if connection.pic_connection_matrix is None:
            continue
        connection.contact_count = edge_support
        joinPieces(connection, segment_list, original_size)
        updatePieceSegmentMap(piece_to_segment, connection.own_segment)
        if on_join is not None:
            on_join(connection, rounds)
        rounds += 1

    if fallback_pairwise and len(segment_list) > 1:
        rounds += assembleGallagherPairwiseKruskal(
            segment_list,
            original_size,
            top_candidates_per_edge=top_candidates_per_edge,
            mutual_edges_only=False,
            max_score=max_score,
            on_join=on_join,
        )
    return rounds


def cloneSegmentList(segment_list):
    cloned_segments, _segment_mapping = cloneSegmentListWithMapping(
        segment_list)
    return cloned_segments


def cloneSegmentListWithMapping(segment_list, include_piece_mapping=False):
    if not segment_list:
        if include_piece_mapping:
            return [], {}, {}
        return [], {}

    score_dict = segment_list[0].score_dict
    connections_dict = {}
    cloned_by_piece_number = {}
    segment_mapping = {}
    piece_mapping = {}

    def clone_piece(piece):
        cloned = cloned_by_piece_number.get(piece.piece_number)
        if cloned is not None:
            piece_mapping[piece] = cloned
            return cloned
        cloned = Segment(
            piece.pic_matrix,
            piece.max_width,
            piece.max_height,
            piece.piece_number,
            piece.component_id,
            score_dict,
            connections_dict,
        )
        cloned_by_piece_number[piece.piece_number] = cloned
        piece_mapping[piece] = cloned
        return cloned

    def clone_connection_matrix(matrix):
        cloned_matrix = np.zeros(matrix.shape, dtype=object)
        for index, piece in np.ndenumerate(matrix):
            if piece != 0:
                cloned_matrix[index] = clone_piece(piece)
        return cloned_matrix

    cloned_segments = []
    for segment in segment_list:
        cloned_segment = clone_piece(segment)
        segment_mapping[segment] = cloned_segment
        cloned_segment.component_id = segment.component_id
        cloned_segment.pic_connection_matrix = clone_connection_matrix(
            segment.pic_connection_matrix)
        cloned_segment.binary_connection_matrix = np.array(
            segment.binary_connection_matrix,
            copy=True,
        )
        cloned_segment.best_connection_found_so_far = BestConnection()
        cloned_segment._kruskal_component_data = None
        cloned_segment.enforce_frame_bounds = getattr(
            segment,
            "enforce_frame_bounds",
            True,
        )
        cloned_segments.append(cloned_segment)
    if include_piece_mapping:
        return cloned_segments, segment_mapping, piece_mapping
    return cloned_segments, segment_mapping


def cloneKruskalConnectionForBranch(
        connection,
        child_own_segment,
        child_join_segment,
        piece_mapping):
    child_connection = BestConnection(
        own_segment=child_own_segment,
        join_segment=child_join_segment,
    )
    child_connection.score = connection.score
    child_connection.second_best_score = connection.second_best_score
    child_connection.contact_count = connection.contact_count

    if connection.kruskal_connection_data is not None:
        (
            own_data,
            compare_data,
            compare_row_offset,
            compare_col_offset,
            own_row_offset,
            own_col_offset,
            height_padded,
            width_padded,
        ) = connection.kruskal_connection_data
        child_connection.pic_connection_matrix = (
            cloneKruskalConnectionMatrixForBranch(
                own_data,
                compare_data,
                compare_row_offset,
                compare_col_offset,
                own_row_offset,
                own_col_offset,
                height_padded,
                width_padded,
                piece_mapping,
            )
        )
        child_connection.binary_connection_matrix = (
            kruskalBinaryConnectionMatrix(
                own_data,
                compare_data,
                compare_row_offset,
                compare_col_offset,
                own_row_offset,
                own_col_offset,
                height_padded,
                width_padded,
            )
        )
        return child_connection

    if connection.pic_connection_matrix is not None:
        child_connection.pic_connection_matrix = clonePieceMatrixForBranch(
            connection.pic_connection_matrix,
            piece_mapping,
        )
        child_connection.binary_connection_matrix = np.array(
            connection.binary_connection_matrix,
            copy=True,
        )
    return child_connection


def cloneKruskalConnectionMatrixForBranch(
        own_data,
        compare_data,
        compare_row_offset,
        compare_col_offset,
        own_row_offset,
        own_col_offset,
        height_padded,
        width_padded,
        piece_mapping):
    combined_pointer = np.zeros((height_padded, width_padded), dtype=object)
    for row, col in own_data["positions"]:
        combined_pointer[row + own_row_offset, col + own_col_offset] = (
            piece_mapping[own_data["pic_matrix"][row, col]]
        )
    for row, col in compare_data["positions"]:
        combined_pointer[
            row + compare_row_offset,
            col + compare_col_offset,
        ] = piece_mapping[compare_data["pic_matrix"][row, col]]
    return combined_pointer


def kruskalBinaryConnectionMatrix(
        own_data,
        compare_data,
        compare_row_offset,
        compare_col_offset,
        own_row_offset,
        own_col_offset,
        height_padded,
        width_padded):
    combined_pieces = np.zeros((height_padded, width_padded))
    for row, col in own_data["positions"]:
        combined_pieces[row + own_row_offset, col + own_col_offset] = 1
    for row, col in compare_data["positions"]:
        combined_pieces[row + compare_row_offset, col + compare_col_offset] = 1
    return combined_pieces


def clonePieceMatrixForBranch(matrix, piece_mapping):
    cloned_matrix = np.zeros(matrix.shape, dtype=object)
    for index, piece in np.ndenumerate(matrix):
        if piece != 0:
            cloned_matrix[index] = piece_mapping[piece]
    return cloned_matrix


def findTopConnectionsKruskal(
        segment_list,
        limit,
        compare_type,
        boost_priority_of_big_pieces_joining,
        compare_mode):
    if limit <= 0:
        return []

    counter = itertools.count()
    best_connections = []

    def consider_connection(segment1, segment2):
        segment1.best_connection_found_so_far = BestConnection()
        connection = segment1.calculateConnectionsKruskal(
            segment2,
            boost_priority_of_big_pieces_joining,
            defer_connection_matrices=True,
        )
        if not connection.hasConnection():
            return
        best_connections.append((
            connectionPriority(connection, compare_type),
            next(counter),
            connection,
        ))

    if compare_mode == CompareWithOtherSegments.ONLY_BEST:
        for index, segment1 in enumerate(segment_list):
            for segment2 in segment_list[index+1:]:
                consider_connection(segment1, segment2)
    else:
        for segment1 in segment_list:
            for segment2 in segment_list:
                if segment1 != segment2:
                    consider_connection(segment1, segment2)

    return [
        connection
        for _priority, _counter, connection
        in heapq.nsmallest(limit, best_connections)
    ]


def findSegmentByComponent(segment_list, component_id, piece_number):
    for segment in segment_list:
        if (
                segment.component_id == component_id
                and segment.piece_number == piece_number):
            return segment
    return None


def assembleKruskalBeamSearch(
        segment_list,
        original_size,
        beam_width=2,
        beam_candidates=None,
        boost_priority_of_big_pieces_joining=False,
        compare_type=CompareWithOtherSegments.ONLY_BEST,
        compare_mode=CompareWithOtherSegments.ONLY_BEST,
        show_progress=False,
        on_join=None):
    if beam_width <= 1:
        return assembleKruskalWithPriorityQueue(
            segment_list,
            original_size,
            boost_priority_of_big_pieces_joining,
            compare_type,
            compare_mode,
            on_join,
        )
    if beam_candidates is None:
        beam_candidates = beam_width

    replay_source = cloneSegmentList(segment_list) if on_join is not None else None
    states = [
        KruskalBeamState(
            segment_list,
            connection_queue=KruskalConnectionPriorityQueue(
                segment_list,
                boost_priority_of_big_pieces_joining,
                compare_type,
                compare_mode,
            ),
        )
    ]
    while True:
        child_specs = []
        sequence = itertools.count()

        for state in states:
            if len(state.segment_list) <= 1:
                child_specs.append((
                    state.path_score,
                    next(sequence),
                    state,
                    None,
                ))
                continue

            candidates = state.connection_queue.topConnections(
                state.segment_list,
                beam_candidates,
            )
            for connection in candidates:
                priority = connectionPriority(connection, compare_type)
                child_specs.append((
                    state.path_score + priority,
                    next(sequence),
                    state,
                    connection,
                ))

        if not child_specs:
            break

        selected_specs = heapq.nsmallest(beam_width, child_specs)
        selected_count_by_state = {}
        selected_seen_by_state = {}
        for _path_score, _sequence, state, _connection in selected_specs:
            selected_count_by_state[state] = (
                selected_count_by_state.get(state, 0) + 1
            )
        states = []
        for path_score, _sequence, state, connection in selected_specs:
            if connection is None:
                states.append(state)
                continue

            selected_seen_by_state[state] = (
                selected_seen_by_state.get(state, 0) + 1
            )
            can_reuse_state = (
                selected_seen_by_state[state]
                == selected_count_by_state[state]
            )
            if can_reuse_state:
                child_segments = state.segment_list
                child_own_segment = connection.own_segment
                child_join_segment = connection.join_segment
                child_connection = connection
                if child_connection.pic_connection_matrix is None:
                    child_own_segment.materializeKruskalConnection(
                        child_connection)
                child_queue = state.connection_queue
            else:
                (
                    child_segments,
                    segment_mapping,
                    piece_mapping,
                ) = cloneSegmentListWithMapping(
                    state.segment_list,
                    include_piece_mapping=True,
                )
                child_own_segment = segment_mapping.get(
                    connection.own_segment)
                child_join_segment = segment_mapping.get(
                    connection.join_segment)
                if child_own_segment is None or child_join_segment is None:
                    continue

                child_connection = cloneKruskalConnectionForBranch(
                    connection,
                    child_own_segment,
                    child_join_segment,
                    piece_mapping,
                )
                child_queue = state.connection_queue.cloneForBranch(
                    segment_mapping,
                    (connection.own_segment, connection.join_segment),
                )
            if child_connection.pic_connection_matrix is None:
                continue

            history_item = (
                child_own_segment.component_id,
                child_own_segment.piece_number,
                child_join_segment.component_id,
                child_join_segment.piece_number,
            )
            joinPieces(child_connection, child_segments, original_size)
            child_queue.addConnectionsFor(
                child_connection.own_segment,
                child_segments,
            )
            if on_join is None:
                child_history = ()
            else:
                child_history = state.history + (history_item,)
            child_state = KruskalBeamState(
                child_segments,
                connection_queue=child_queue,
                path_score=path_score,
                rounds=state.rounds + 1,
                history=child_history,
            )
            states.append(child_state)

        if not states:
            break

        best_state = min(
            states,
            key=lambda state: (len(state.segment_list), state.path_score),
        )
        if show_progress:
            print(
                "beam round",
                max(state.rounds for state in states),
                "states",
                len(states),
                "best remaining",
                len(best_state.segment_list),
                "best path score",
                best_state.path_score,
            )
        if all(len(state.segment_list) <= 1 for state in states):
            break

    best_state = min(
        states,
        key=lambda state: (len(state.segment_list), state.path_score),
    )
    if on_join is not None:
        replayBeamHistory(
            replay_source,
            best_state.history,
            original_size,
            boost_priority_of_big_pieces_joining,
            on_join,
        )
    segment_list[:] = best_state.segment_list
    return best_state.rounds


def assembleKruskalHybridBeamSearch(
        segment_list,
        original_size,
        beam_start_components,
        beam_width=2,
        beam_candidates=None,
        boost_priority_of_big_pieces_joining=False,
        compare_type=CompareWithOtherSegments.ONLY_BEST,
        compare_mode=CompareWithOtherSegments.ONLY_BEST,
        show_progress=False,
        on_join=None):
    if beam_start_components is None or len(segment_list) <= beam_start_components:
        return assembleKruskalBeamSearch(
            segment_list,
            original_size,
            beam_width,
            beam_candidates,
            boost_priority_of_big_pieces_joining,
            compare_type,
            compare_mode,
            show_progress,
            on_join,
        )

    connection_queue = KruskalConnectionPriorityQueue(
        segment_list,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
    )
    rounds = 0
    while len(segment_list) > beam_start_components:
        best_connection = connection_queue.popBestConnection(segment_list)
        if best_connection.pic_connection_matrix is None:
            return rounds
        joinPieces(best_connection, segment_list, original_size)
        connection_queue.addConnectionsFor(
            best_connection.own_segment,
            segment_list,
        )
        if on_join is not None:
            on_join(best_connection, rounds)
        rounds += 1

    if len(segment_list) <= 1:
        return rounds

    def offsetJoinRound(best_connection, beam_round):
        if on_join is not None:
            on_join(best_connection, rounds + beam_round)

    beam_rounds = assembleKruskalBeamSearch(
        segment_list,
        original_size,
        beam_width,
        beam_candidates,
        boost_priority_of_big_pieces_joining,
        compare_type,
        compare_mode,
        show_progress,
        offsetJoinRound,
    )
    return rounds + beam_rounds


def replayBeamHistory(
        segment_list,
        history,
        original_size,
        boost_priority_of_big_pieces_joining,
        on_join):
    replay_segments = cloneSegmentList(segment_list)
    for round_number, (
            own_component_id,
            own_piece_number,
            join_component_id,
            join_piece_number,
    ) in enumerate(history):
        own_segment = findSegmentByComponent(
            replay_segments,
            own_component_id,
            own_piece_number,
        )
        join_segment = findSegmentByComponent(
            replay_segments,
            join_component_id,
            join_piece_number,
        )
        if own_segment is None or join_segment is None:
            break
        own_segment.best_connection_found_so_far = BestConnection()
        connection = own_segment.calculateConnectionsKruskal(
            join_segment,
            boost_priority_of_big_pieces_joining,
        )
        if connection.pic_connection_matrix is None:
            break
        joinPieces(connection, replay_segments, original_size)
        on_join(connection, round_number)


def findBestConnectionPrim(segment_list, root_segment, compare_type):
    best_so_far = BestConnection()
    for segment in segment_list:
        if segment != root_segment:
            root_segment.best_connection_found_so_far = BestConnection()
            temp = root_segment.calculateConnectionsPrim(segment)
            if temp.isBetterConnection(best_so_far, compare_type):
                best_so_far = temp
    return best_so_far


def findBestRootSegment(segment_list):
    return findBestPrimSeedSegment(segment_list, strategy="random")


def findBestBuddyConnection(
        segment,
        segment_list,
        single_piece_by_segment=None,
        score_values=None):
    if single_piece_by_segment is None:
        segment_is_single_piece = isSinglePiece(segment)
    else:
        segment_is_single_piece = single_piece_by_segment[segment]
    if score_values is None and segment_is_single_piece:
        score_values = scoreValuesArray(segment.score_dict)
    best_so_far = BestConnection()
    for segment2 in segment_list:
        if segment != segment2:
            if single_piece_by_segment is None:
                segment2_is_single_piece = isSinglePiece(segment2)
            else:
                segment2_is_single_piece = single_piece_by_segment[segment2]
            if segment_is_single_piece and segment2_is_single_piece:
                if score_values is None:
                    best_direction, score, second_best_score = (
                        singlePieceBestDirection(segment, segment2)
                    )
                else:
                    best_direction, score, second_best_score = (
                        singlePieceBestDirectionDense(
                            segment,
                            segment2,
                            score_values,
                        )
                    )
                if score < best_so_far.score:
                    best_so_far = buildSinglePieceConnection(
                        segment,
                        segment2,
                        best_direction,
                        score,
                        second_best_score,
                    )
            else:
                segment.best_connection_found_so_far = BestConnection()
                temp = segment.calculateConnectionsKruskal(segment2, False)
                if temp.isBetterConnection(
                        best_so_far, CompareWithOtherSegments.ONLY_BEST):
                    best_so_far = temp
    return best_so_far


def isSinglePiece(segment):
    binary_connection_matrix = getattr(segment, "binary_connection_matrix", None)
    if binary_connection_matrix is None:
        return False
    return np.count_nonzero(binary_connection_matrix) == 1


def calculateSinglePieceConnection(segment, compare_segment):
    best_direction, best_score, second_best_score = singlePieceBestDirection(
        segment,
        compare_segment,
    )
    return buildSinglePieceConnection(
        segment,
        compare_segment,
        best_direction,
        best_score,
        second_best_score,
    )


def singlePieceBestDirection(segment, compare_segment):
    score_dict = segment.score_dict
    score_values = scoreValuesArray(score_dict)
    if score_values is not None:
        return singlePieceBestDirectionDense(
            segment,
            compare_segment,
            score_values,
        )

    segment_piece_number = segment.piece_number
    compare_piece_number = compare_segment.piece_number
    best_direction = None
    best_score = None
    second_best_score = float("inf")

    for direction in JoinDirection:
        score = score_dict[
            segment_piece_number,
            direction,
            compare_piece_number,
        ]
        if best_score is None or score < best_score:
            if best_score is not None:
                second_best_score = best_score
            best_direction = direction
            best_score = score
        elif score < second_best_score:
            second_best_score = score

    return best_direction, best_score, second_best_score


def singlePieceBestDirectionDense(segment, compare_segment, score_values):
    segment_piece_number = segment.piece_number
    compare_piece_number = compare_segment.piece_number
    scores = score_values[segment_piece_number, :, compare_piece_number]
    best_direction = None
    best_score = None
    second_best_score = float("inf")

    for direction_index, direction in enumerate(DIRECTIONS_BY_INDEX):
        score = float(scores[direction_index])
        if best_score is None or score < best_score:
            if best_score is not None:
                second_best_score = best_score
            best_direction = direction
            best_score = score
        elif score < second_best_score:
            second_best_score = score

    return best_direction, best_score, second_best_score


def buildSinglePieceConnection(
        segment,
        compare_segment,
        best_direction,
        best_score,
        second_best_score):
    if best_direction == JoinDirection.UP:
        pic_connection_matrix = np.asarray(
            [[compare_segment], [segment]], dtype=object)
        binary_connection_matrix = np.asarray([[1], [1]])
    elif best_direction == JoinDirection.DOWN:
        pic_connection_matrix = np.asarray(
            [[segment], [compare_segment]], dtype=object)
        binary_connection_matrix = np.asarray([[1], [1]])
    elif best_direction == JoinDirection.LEFT:
        pic_connection_matrix = np.asarray(
            [[compare_segment, segment]], dtype=object)
        binary_connection_matrix = np.asarray([[1, 1]])
    else:
        pic_connection_matrix = np.asarray(
            [[segment, compare_segment]], dtype=object)
        binary_connection_matrix = np.asarray([[1, 1]])

    connection = BestConnection(
        own_segment=segment,
        pic_connection_matrix=pic_connection_matrix,
        join_segment=compare_segment,
        binary_connection_matrix=binary_connection_matrix,
    )
    connection.score = best_score
    connection.second_best_score = second_best_score
    connection.contact_count = 1
    return connection


def connectBestBudsFirst(segment_list, original_size, show_progress=True):
    candidates = list(segment_list)
    single_piece_by_segment = {
        segment: isSinglePiece(segment)
        for segment in candidates
    }
    best_by_segment = {
        segment: findBestBuddyConnection(
            segment,
            candidates,
            single_piece_by_segment,
        )
        for segment in candidates
    }
    active_segments = set(segment_list)
    for segment1 in candidates:
        if segment1 not in active_segments:
            continue
        best_so_far = best_by_segment[segment1]
        if best_so_far.join_segment not in active_segments:
            continue
        best_so_far2 = best_by_segment[best_so_far.join_segment]
        is_mutual_best_match = (
            best_so_far.own_segment.piece_number == best_so_far2.join_segment.piece_number
            and best_so_far.join_segment.piece_number == best_so_far2.own_segment.piece_number
        )
        if show_progress:
            status = "mutual match" if is_mutual_best_match else "checked"
            print(
                "Best-buddy check: "
                f"{best_so_far.own_segment.piece_number} -> {best_so_far.join_segment.piece_number}; "
                f"{best_so_far2.own_segment.piece_number} -> {best_so_far2.join_segment.piece_number} "
                f"({status})"
            )
        if is_mutual_best_match:
            joinPieces(best_so_far2, segment_list, original_size)
            active_segments.remove(best_so_far2.join_segment)


def joinPieces(best_connection, segment_list, original_size):
    best_connection.own_segment.materializeKruskalConnection(best_connection)
    best_connection.stripZeros()
    best_connection.own_segment.binary_connection_matrix = best_connection.binary_connection_matrix
    best_connection.own_segment.pic_connection_matrix = best_connection.pic_connection_matrix
    best_connection.own_segment.component_id += original_size
    segment_list.remove(best_connection.join_segment)
