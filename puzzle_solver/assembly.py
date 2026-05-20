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


def printPiecesMatrices(segment_list):
    for node in segment_list:
        print(node.binary_connection_matrix)
        print(node.pic_connection_matrix)
        print(node.piece_number)
    print('\n\n\n')


def clearDictionaryForRam(my_list, removing):
    for connection in my_list:
        for key in dict(connection.connections_dict):
            if key[0] == removing or key[1] == removing:
                del connection.connections_dict[key]


def createCrossPiece(segment_list):
    root = findBestRootSegment(segment_list)


def checkFunctionCalculatesTheSameOnEachPiece(segment_list, boost_priority_of_big_pieces_joining):
    for segment in segment_list:
        for segment2 in segment_list:
            if segment != segment2:
                segment.best_connection_found_so_far = BestConnection()
                segment2.best_connection_found_so_far = BestConnection()
                print(segment.best_connection_found_so_far)
                temp1 = segment.calculateConnectionsKruskal(
                    segment2, boost_priority_of_big_pieces_joining)
                temp2 = segment2.calculateConnectionsKruskal(
                    segment, boost_priority_of_big_pieces_joining)
                print(temp1 == temp2)


def findBestBuddyConnection(segment, segment_list):
    segment_is_single_piece = isSinglePiece(segment)
    best_so_far = BestConnection()
    for segment2 in segment_list:
        if segment != segment2:
            if segment_is_single_piece and isSinglePiece(segment2):
                temp = calculateSinglePieceConnection(segment, segment2)
            else:
                segment.best_connection_found_so_far = BestConnection()
                temp = segment.calculateConnectionsKruskal(segment2, False)
            if temp.isBetterConnection(best_so_far, CompareWithOtherSegments.ONLY_BEST):
                best_so_far = temp
    return best_so_far


def isSinglePiece(segment):
    binary_connection_matrix = getattr(segment, "binary_connection_matrix", None)
    if binary_connection_matrix is None:
        return False
    return np.count_nonzero(binary_connection_matrix) == 1


def calculateSinglePieceConnection(segment, compare_segment):
    direction_scores = [
        (
            direction,
            segment.score_dict[
                segment.piece_number,
                direction,
                compare_segment.piece_number,
            ],
        )
        for direction in JoinDirection
    ]
    direction_scores.sort(key=lambda item: item[1])
    best_direction, best_score = direction_scores[0]
    second_best_score = (
        direction_scores[1][1]
        if len(direction_scores) > 1
        else best_score
    )

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
    best_by_segment = {
        segment: findBestBuddyConnection(segment, candidates)
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
