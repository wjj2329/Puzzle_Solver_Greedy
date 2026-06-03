from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

import numpy as np

from .distances import (
    euclideanDistances,
    mahalanobisEdgeDistances,
    mgcEdgeDistances,
)
from .enums import ScoreAlgorithm, ScoreMode
from .enums import OPPOSITE_DIRECTIONS
from .models import ScorePayload
from .score_helpers import (
    JOIN_EDGE_PAIR_INDICES,
    scoreComponentCount,
    scorePayloadPair,
)


_SCORE_PAYLOADS = None
_SCORE_ALGORITHM = None
_SCORE_PAYLOAD_ARRAYS = None


def initializeScoreWorker(score_payloads, score_algorithm):
    global _SCORE_PAYLOADS, _SCORE_ALGORITHM, _SCORE_PAYLOAD_ARRAYS
    _SCORE_PAYLOADS = score_payloads
    _SCORE_ALGORITHM = score_algorithm
    _SCORE_PAYLOAD_ARRAYS = None


def scoreEntriesForPayloadIndex(index):
    entries = []
    segment1 = _SCORE_PAYLOADS[index]
    for segment2 in _SCORE_PAYLOADS[index+1:]:
        entries.extend(scorePayloadPair(segment1, segment2, _SCORE_ALGORITHM))
    return entries


def scoreEntriesForPayloadRange(start, stop):
    entries = []
    for index in range(start, stop):
        entries.extend(scoreEntriesForPayloadIndex(index))
    return entries


def scoreArraysForPayloadIndex(index):
    return scoreArraysForPayloadRange(index, index + 1)


def scoreArrayEntryCount(start, stop):
    payload_count = len(_SCORE_PAYLOADS)
    return sum(
        max(payload_count - index - 1, 0) * 8
        for index in range(start, stop)
    )


def scorePayloadArrayCache():
    global _SCORE_PAYLOAD_ARRAYS
    if _SCORE_PAYLOAD_ARRAYS is None:
        _SCORE_PAYLOAD_ARRAYS = buildScorePayloadArrayCache(_SCORE_PAYLOADS)
    return _SCORE_PAYLOAD_ARRAYS


def buildScorePayloadArrayCache(score_payloads):
    return {
        "piece_numbers": np.asarray(
            [payload.piece_number for payload in score_payloads],
            dtype=np.int32,
        ),
        "own": buildEdgeArrayCache(score_payloads, "own_edges"),
        "compare": buildEdgeArrayCache(score_payloads, "compare_edges"),
    }


def buildEdgeArrayCache(score_payloads, edge_attribute):
    edge_cache = {}
    for direction, _compare_direction, _direction_index, _opposite_index in (
            JOIN_EDGE_PAIR_INDICES):
        if direction in edge_cache:
            continue
        edges = [
            getattr(payload, edge_attribute)[direction]
            for payload in score_payloads
        ]
        edge_cache[direction] = {
            "edge": np.asarray([edge.edge for edge in edges], dtype=np.float64),
            "average_delta": np.asarray(
                [edge.average_delta for edge in edges],
                dtype=np.float64,
            ),
            "inverse_covariance": np.asarray(
                [edge.inverse_covariance for edge in edges],
                dtype=np.float64,
            ),
            "gradient_average": np.asarray(
                [edge.gradient_average for edge in edges],
                dtype=np.float64,
            ),
            "gradient_inverse_covariance": np.asarray(
                [edge.gradient_inverse_covariance for edge in edges],
                dtype=np.float64,
            ),
        }
    return edge_cache


def scoreArraysForPayloadRange(start, stop):
    payload_arrays = scorePayloadArrayCache()
    component_count = scoreComponentCount(_SCORE_ALGORITHM)
    entry_count = scoreArrayEntryCount(start, stop)
    own_numbers = np.empty(entry_count, dtype=np.int32)
    direction_indices = np.empty(entry_count, dtype=np.int8)
    join_numbers = np.empty(entry_count, dtype=np.int32)
    score_values = np.empty((entry_count, component_count), dtype=np.float64)

    cursor = 0
    for index in range(start, stop):
        cursor = writeScoreArraysForPayloadIndex(
            payload_arrays,
            index,
            _SCORE_ALGORITHM,
            own_numbers,
            direction_indices,
            join_numbers,
            score_values,
            cursor,
        )

    return (
        own_numbers[:cursor],
        direction_indices[:cursor],
        join_numbers[:cursor],
        score_values[:cursor],
        component_count == 1,
    )


def writeScoreArraysForPayloadIndex(
        payload_arrays,
        index,
        score_algorithm,
        own_numbers,
        direction_indices,
        join_numbers,
        score_values,
        cursor):
    piece_numbers = payload_arrays["piece_numbers"]
    remaining_start = index + 1
    if remaining_start >= len(piece_numbers):
        return cursor

    join_piece_numbers = piece_numbers[remaining_start:]
    own_number = piece_numbers[index]
    for (
            own_direction,
            compare_direction,
            direction_index,
            opposite_direction_index) in JOIN_EDGE_PAIR_INDICES:
        own_edges = payload_arrays["own"][own_direction]
        compare_edges = payload_arrays["compare"][compare_direction]
        component_scores = scoreComponentsForDirection(
            score_algorithm,
            own_edges,
            compare_edges,
            index,
            remaining_start,
        )
        cursor = writeReciprocalScoreArrays(
            own_numbers,
            direction_indices,
            join_numbers,
            score_values,
            cursor,
            own_number,
            join_piece_numbers,
            direction_index,
            opposite_direction_index,
            component_scores,
        )
    return cursor


def scoreComponentsForDirection(
        score_algorithm,
        own_edges,
        compare_edges,
        index,
        remaining_start):
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN:
        return (
            euclideanDistances(
                own_edges["edge"][index],
                compare_edges["edge"][remaining_start:],
            ),
        )
    if score_algorithm == ScoreAlgorithm.MAHALANOBIS:
        return (
            mahalanobisEdgeDistances(
                own_edges["edge"][index],
                own_edges["average_delta"][index],
                own_edges["inverse_covariance"][index],
                compare_edges["edge"][remaining_start:],
                compare_edges["average_delta"][remaining_start:],
                compare_edges["inverse_covariance"][remaining_start:],
            ),
        )
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        return (
            mahalanobisEdgeDistances(
                own_edges["edge"][index],
                own_edges["average_delta"][index],
                own_edges["inverse_covariance"][index],
                compare_edges["edge"][remaining_start:],
                compare_edges["average_delta"][remaining_start:],
                compare_edges["inverse_covariance"][remaining_start:],
            ),
            euclideanDistances(
                own_edges["edge"][index],
                compare_edges["edge"][remaining_start:],
            ),
        )
    if score_algorithm in (ScoreAlgorithm.MGC, ScoreAlgorithm.MGC_DISTANCE):
        return (
            mgcEdgeDistances(
                own_edges["edge"][index],
                own_edges["gradient_average"][index],
                own_edges["gradient_inverse_covariance"][index],
                compare_edges["edge"][remaining_start:],
                compare_edges["gradient_average"][remaining_start:],
                compare_edges["gradient_inverse_covariance"][remaining_start:],
                sqrt=score_algorithm == ScoreAlgorithm.MGC_DISTANCE,
            ),
        )
    return ()


def writeReciprocalScoreArrays(
        own_numbers,
        direction_indices,
        join_numbers,
        score_values,
        cursor,
        own_number,
        join_piece_numbers,
        direction_index,
        opposite_direction_index,
        component_scores):
    candidate_count = len(join_piece_numbers)
    stop = cursor + candidate_count
    own_numbers[cursor:stop] = own_number
    direction_indices[cursor:stop] = direction_index
    join_numbers[cursor:stop] = join_piece_numbers
    for component_index, scores in enumerate(component_scores):
        score_values[cursor:stop, component_index] = scores
    cursor = stop

    stop = cursor + candidate_count
    own_numbers[cursor:stop] = join_piece_numbers
    direction_indices[cursor:stop] = opposite_direction_index
    join_numbers[cursor:stop] = own_number
    for component_index, scores in enumerate(component_scores):
        score_values[cursor:stop, component_index] = scores
    return stop


def chunkRanges(length, max_chunks):
    if length <= 0:
        return []
    chunk_count = min(length, max_chunks)
    chunk_size = (length + chunk_count - 1) // chunk_count
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def scoreEntriesForPair(segment1, segment2, score_algorithm):
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN:
        return segment1.scoreEntriesEuclidean(segment2)
    elif score_algorithm == ScoreAlgorithm.MAHALANOBIS:
        return segment1.scoreEntriesMahalanobis(segment2)
    elif score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        return segment1.scoreEntriesEuclideanAndMahalanobis(segment2)
    elif score_algorithm == ScoreAlgorithm.MGC:
        return segment1.scoreEntriesMGC(segment2)
    elif score_algorithm == ScoreAlgorithm.MGC_DISTANCE:
        return segment1.scoreEntriesMGCDistance(segment2)
    return None


def scoreEntriesForSegment(segment1, remaining_segments, score_algorithm):
    entries = []
    for segment2 in remaining_segments:
        entries.extend(scoreEntriesForPair(segment1, segment2, score_algorithm))
    return entries


def calculateScoresSerial(segment_list, score_algorithm, show_progress=True):
    for index, segment1 in enumerate(segment_list):
        if show_progress:
            print("calculating score for segment ", segment1.piece_number)
        for segment2 in segment_list[index+1:]:
            if score_algorithm == ScoreAlgorithm.EUCLIDEAN:
                segment1.calculateScoreEuclidean(segment2)
            elif score_algorithm == ScoreAlgorithm.MAHALANOBIS:
                segment1.calculateScoreMahalanobis(segment2)
            elif score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
                segment1.calculateScoreEuclideanAndMahalanobis(segment2)
            elif score_algorithm == ScoreAlgorithm.MGC:
                segment1.calculateScoreMGC(segment2)
            elif score_algorithm == ScoreAlgorithm.MGC_DISTANCE:
                segment1.calculateScoreMGCDistance(segment2)


def precomputeScoreEdges(segment_list):
    for segment in segment_list:
        segment.ownScoreEdges()
        segment.compareScoreEdges()


def buildScorePayloads(segment_list):
    precomputeScoreEdges(segment_list)
    return tuple(
        ScorePayload(
            segment.piece_number,
            segment.ownScoreEdges(),
            segment.compareScoreEdges(),
        )
        for segment in segment_list
    )


def calculateScoresThreaded(segment_list, score_algorithm, show_progress=True, max_workers=None):
    if len(segment_list) < 2:
        return
    if max_workers is None:
        max_workers = min(len(segment_list), os.cpu_count() or 1)
    if max_workers <= 1:
        calculateScoresSerial(segment_list, score_algorithm, show_progress)
        return

    precomputeScoreEdges(segment_list)
    score_dict = segment_list[0].score_dict
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, segment1 in enumerate(segment_list):
            if show_progress:
                print("calculating score for segment ", segment1.piece_number)
            futures.append(executor.submit(
                scoreEntriesForSegment,
                segment1,
                tuple(segment_list[index+1:]),
                score_algorithm,
            ))
        for future in as_completed(futures):
            entries = future.result()
            if hasattr(score_dict, "setMany"):
                score_dict.setMany(entries)
            else:
                for key, score in entries:
                    score_dict[key] = score


def calculateScoresProcess(segment_list, score_algorithm, show_progress=True, max_workers=None):
    if len(segment_list) < 2:
        return
    if max_workers is None:
        max_workers = min(len(segment_list), os.cpu_count() or 1)
    if max_workers <= 1:
        calculateScoresSerial(segment_list, score_algorithm, show_progress)
        return

    score_payloads = buildScorePayloads(segment_list)
    score_dict = segment_list[0].score_dict
    use_score_arrays = hasattr(score_dict, "setManyArrays")
    score_worker = (
        scoreArraysForPayloadRange
        if use_score_arrays
        else scoreEntriesForPayloadRange
    )
    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializeScoreWorker,
            initargs=(score_payloads, score_algorithm)) as executor:
        futures = []
        for start, stop in chunkRanges(len(segment_list), max_workers * 4):
            if show_progress:
                print("calculating score for segments ",
                      segment_list[start].piece_number, " through ",
                      segment_list[stop - 1].piece_number)
            futures.append(executor.submit(score_worker, start, stop))
        for future in as_completed(futures):
            result = future.result()
            if use_score_arrays:
                score_dict.setManyArrays(*result)
            elif hasattr(score_dict, "setMany"):
                entries = result
                score_dict.setMany(entries)
            else:
                entries = result
                for key, score in entries:
                    score_dict[key] = score


def calculateScores(segment_list, score_algorithm, show_progress=True, max_workers=None, executor_type="thread"):
    if executor_type == "serial":
        calculateScoresSerial(segment_list, score_algorithm, show_progress)
    elif executor_type == "process":
        calculateScoresProcess(segment_list, score_algorithm, show_progress, max_workers)
    elif executor_type == "thread":
        calculateScoresThreaded(segment_list, score_algorithm, show_progress, max_workers)
    else:
        raise ValueError("executor_type must be 'serial', 'thread', or 'process'")


def scoreDict(segment_list):
    if not segment_list:
        return {}
    return segment_list[0].score_dict


def finalizeScores(
        segment_list,
        score_algorithm,
        score_mode=ScoreMode.DISSIMILARITY):
    normalizeScores(segment_list, score_algorithm)
    applyScoreMode(segment_list, score_mode)


def applyScoreMode(segment_list, score_mode=ScoreMode.DISSIMILARITY):
    if score_mode == ScoreMode.DISSIMILARITY:
        return
    if score_mode == ScoreMode.RELIABILITY:
        applyReliabilityScores(segment_list)
        return
    raise ValueError("score_mode must be ScoreMode.DISSIMILARITY or ScoreMode.RELIABILITY")


def applyReliabilityScores(segment_list):
    score_dict = scoreDict(segment_list)
    if hasattr(score_dict, "applyReliabilityScores"):
        score_dict.applyReliabilityScores()
        return

    best_by_piece_direction = {}
    second_best_by_piece_direction = {}
    score_count_by_piece_direction = {}
    for own_number, direction, join_number in score_dict:
        key = (own_number, direction)
        score = score_dict[own_number, direction, join_number]
        best_score = best_by_piece_direction.get(key, float("inf"))
        second_best_score = second_best_by_piece_direction.get(key, float("inf"))
        score_count_by_piece_direction[key] = (
            score_count_by_piece_direction.get(key, 0) + 1
        )
        if score <= best_score:
            second_best_score = best_score
            best_score = score
        elif score < second_best_score:
            second_best_score = score
        best_by_piece_direction[key] = best_score
        second_best_by_piece_direction[key] = second_best_score

    for own_number, direction, join_number in list(score_dict):
        raw_score = score_dict[own_number, direction, join_number]
        key = (own_number, direction)
        if score_count_by_piece_direction[key] < 2:
            second_best_score = best_by_piece_direction[key]
        else:
            second_best_score = second_best_by_piece_direction[key]
        score_dict[own_number, direction, join_number] = reliabilityScore(
            raw_score,
            second_best_score,
        )


def reliabilityScore(score, second_best_score):
    if second_best_score <= 0:
        if score <= 0:
            return 1.0
        return float("inf")
    return score / second_best_score


def applySymmetricCompatibilityScores(segment_list):
    score_dict = scoreDict(segment_list)
    if hasattr(score_dict, "applySymmetricCompatibilityScores"):
        score_dict.applySymmetricCompatibilityScores()
        return

    updates = {}
    for own_number, direction, join_number in list(score_dict):
        reciprocal_key = (
            join_number,
            OPPOSITE_DIRECTIONS[direction],
            own_number,
        )
        if reciprocal_key not in score_dict:
            continue
        key = (own_number, direction, join_number)
        if key in updates:
            continue
        score = score_dict[key]
        reciprocal_score = score_dict[reciprocal_key]
        symmetric_score = (score + reciprocal_score) / 2.0
        updates[key] = symmetric_score
        updates[reciprocal_key] = symmetric_score
    for key, score in updates.items():
        score_dict[key] = score


def normalizeScores(segment_list, score_algorithm):
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        score_dict = scoreDict(segment_list)
        if hasattr(score_dict, "normalizeCombinedScores"):
            score_dict.normalizeCombinedScores()
            return

        list1 = []
        list2 = []
        for value in score_dict.values():
            list1.append(value[0])
            list2.append(value[1])
        max1 = max(list1)
        max2 = max(list2)
        min1 = min(list1)
        min2 = min(list2)
        for value in score_dict:
            mahalanobis_score = score_dict[value][0]
            euclidean_score = score_dict[value][1]
            normalized_mahalanobis_score = (mahalanobis_score-min1)/(max1-min1)
            normalized_euclidean_score = (euclidean_score-min2) / (max2-min2)
            score_dict[value] = normalized_mahalanobis_score+normalized_euclidean_score
