import heapq
import itertools
import random

import numpy as np

from .enums import CompareWithOtherSegments, JoinDirection
from .models import BestConnection, Segment


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
            compare_mode=CompareWithOtherSegments.ONLY_BEST):
        self.boost_priority_of_big_pieces_joining = boost_priority_of_big_pieces_joining
        self.compare_type = compare_type
        self.compare_mode = compare_mode
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
        heap_copy = list(self._heap)
        connections = []
        while heap_copy and len(connections) < limit:
            item = heapq.heappop(heap_copy)
            connection = self._connectionFromItem(
                item,
                active_segments,
                materialize=False,
            )
            if connection is not None:
                connections.append(connection)
        return connections

    def cloneForBranch(self, segment_mapping, excluded_segments):
        cloned = self.__class__.__new__(self.__class__)
        cloned.boost_priority_of_big_pieces_joining = (
            self.boost_priority_of_big_pieces_joining
        )
        cloned.compare_type = self.compare_type
        cloned.compare_mode = self.compare_mode
        cloned._counter = itertools.count()
        cloned._heap = []

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

            cloned_own_segment = segment_mapping.get(own_segment)
            cloned_join_segment = segment_mapping.get(join_segment)
            if cloned_own_segment is None or cloned_join_segment is None:
                continue
            heapq.heappush(
                cloned._heap,
                (
                    priority,
                    next(cloned._counter),
                    None,
                    cloned_own_segment,
                    cloned_join_segment,
                    cloned_own_segment.component_id,
                    cloned_join_segment.component_id,
                ),
            )
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


def cloneSegmentList(segment_list):
    cloned_segments, _segment_mapping = cloneSegmentListWithMapping(
        segment_list)
    return cloned_segments


def cloneSegmentListWithMapping(segment_list):
    if not segment_list:
        return [], {}

    score_dict = segment_list[0].score_dict
    connections_dict = {}
    cloned_by_piece_number = {}
    segment_mapping = {}

    def clone_piece(piece):
        cloned = cloned_by_piece_number.get(piece.piece_number)
        if cloned is not None:
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
        cloned_segments.append(cloned_segment)
    return cloned_segments, segment_mapping


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
        child_states = []
        sequence = itertools.count()

        for state in states:
            if len(state.segment_list) <= 1:
                child_states.append((state.path_score, next(sequence), state))
                continue

            candidates = state.connection_queue.topConnections(
                state.segment_list,
                beam_candidates,
            )
            for connection in candidates:
                child_segments, segment_mapping = cloneSegmentListWithMapping(
                    state.segment_list)
                child_own_segment = segment_mapping.get(
                    connection.own_segment)
                child_join_segment = segment_mapping.get(
                    connection.join_segment)
                if child_own_segment is None or child_join_segment is None:
                    continue

                child_own_segment.best_connection_found_so_far = (
                    BestConnection())
                child_connection = child_own_segment.calculateConnectionsKruskal(
                    child_join_segment,
                    boost_priority_of_big_pieces_joining,
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
                child_queue = state.connection_queue.cloneForBranch(
                    segment_mapping,
                    (connection.own_segment, connection.join_segment),
                )
                child_queue.addConnectionsFor(
                    child_connection.own_segment,
                    child_segments,
                )
                priority = connectionPriority(child_connection, compare_type)
                child_state = KruskalBeamState(
                    child_segments,
                    connection_queue=child_queue,
                    path_score=state.path_score + priority,
                    rounds=state.rounds + 1,
                    history=state.history + (history_item,),
                )
                child_states.append((
                    child_state.path_score,
                    next(sequence),
                    child_state,
                ))

        if not child_states:
            break

        states = [
            state
            for _score, _sequence, state
            in heapq.nsmallest(beam_width, child_states)
        ]
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
            segment_list,
            best_state.history,
            original_size,
            boost_priority_of_big_pieces_joining,
            on_join,
        )
    segment_list[:] = best_state.segment_list
    return best_state.rounds


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
    return random.choice(segment_list)


def findBestBuddyConnection(segment, segment_list, single_piece_by_segment=None):
    if single_piece_by_segment is None:
        segment_is_single_piece = isSinglePiece(segment)
    else:
        segment_is_single_piece = single_piece_by_segment[segment]
    best_so_far = BestConnection()
    for segment2 in segment_list:
        if segment != segment2:
            if single_piece_by_segment is None:
                segment2_is_single_piece = isSinglePiece(segment2)
            else:
                segment2_is_single_piece = single_piece_by_segment[segment2]
            if segment_is_single_piece and segment2_is_single_piece:
                best_direction, score, second_best_score = (
                    singlePieceBestDirection(segment, segment2)
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
