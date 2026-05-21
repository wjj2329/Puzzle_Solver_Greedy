from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

import numpy as np

from .enums import ScoreAlgorithm, ScoreMode
from .models import ScorePayload
from .score_helpers import (
    appendScorePayloadPairArrayValues,
    scoreComponentCount,
    scorePayloadPair,
)


_SCORE_PAYLOADS = None
_SCORE_ALGORITHM = None


def initializeScoreWorker(score_payloads, score_algorithm):
    global _SCORE_PAYLOADS, _SCORE_ALGORITHM
    _SCORE_PAYLOADS = score_payloads
    _SCORE_ALGORITHM = score_algorithm


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


def scoreArraysForPayloadRange(start, stop):
    component_count = scoreComponentCount(_SCORE_ALGORITHM)
    own_numbers = []
    direction_indices = []
    join_numbers = []
    if component_count == 1:
        scores = []
    else:
        scores = ([], [])

    cursor = 0
    for index in range(start, stop):
        segment1 = _SCORE_PAYLOADS[index]
        for segment2 in _SCORE_PAYLOADS[index+1:]:
            cursor = appendScorePayloadPairArrayValues(
                segment1,
                segment2,
                _SCORE_ALGORITHM,
                own_numbers,
                direction_indices,
                join_numbers,
                scores,
                cursor,
            )

    own_numbers = np.asarray(own_numbers, dtype=np.int32)
    direction_indices = np.asarray(direction_indices, dtype=np.int8)
    join_numbers = np.asarray(join_numbers, dtype=np.int32)
    if component_count == 1:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    else:
        score_values = np.empty(
            (len(own_numbers), component_count),
            dtype=np.float64,
        )
        score_values[:, 0] = scores[0]
        score_values[:, 1] = scores[1]

    return (
        own_numbers,
        direction_indices,
        join_numbers,
        score_values,
        component_count == 1,
    )


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
