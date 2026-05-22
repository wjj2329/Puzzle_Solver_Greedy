import heapq
import itertools
import random

import numpy as np

from .enums import CompareWithOtherSegments, JoinDirection
from .models import BestConnection


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
            _, _, connection, own_segment, join_segment, own_component_id, join_component_id = heapq.heappop(self._heap)
            if own_segment not in active_segments or join_segment not in active_segments:
                continue
            if own_segment.component_id != own_component_id:
                continue
            if join_segment.component_id != join_component_id:
                continue
            return connection
        return BestConnection()

    def _pushConnection(self, segment1, segment2):
        segment1.best_connection_found_so_far = BestConnection()
        connection = segment1.calculateConnectionsKruskal(
            segment2,
            self.boost_priority_of_big_pieces_joining,
        )
        if connection.pic_connection_matrix is None:
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
        if self.compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
            return (
                2 * (connection.score * (
                    connection.score / connection.second_best_score
                ))
            ) + connection.score
        return connection.score


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
    best_connection.stripZeros()
    best_connection.own_segment.binary_connection_matrix = best_connection.binary_connection_matrix
    best_connection.own_segment.pic_connection_matrix = best_connection.pic_connection_matrix
    best_connection.own_segment.component_id += original_size
    segment_list.remove(best_connection.join_segment)
