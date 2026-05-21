from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

from .enums import ScoreAlgorithm, ScoreMode
from .models import ScorePayload
from .score_helpers import scorePayloadPair


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
            elif score_algorithm == ScoreAlgorithm.GIST_AND_EUCLIDEAN:
                segment1.calculateScoreGIST(segment2)
            elif score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
                segment1.calculateScoreEuclideanAndMahalanobis(segment2)


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
            for key, score in future.result():
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
            futures.append(executor.submit(scoreEntriesForPayloadRange, start, stop))
        for future in as_completed(futures):
            for key, score in future.result():
                score_dict[key] = score


def calculateScores(segment_list, score_algorithm, show_progress=True, max_workers=None, executor_type="thread"):
    if score_algorithm == ScoreAlgorithm.GIST_AND_EUCLIDEAN:
        calculateScoresSerial(segment_list, score_algorithm, show_progress)
    elif executor_type == "serial":
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
    grouped_scores = defaultdict(list)
    for own_number, direction, join_number in score_dict:
        grouped_scores[own_number, direction].append(
            score_dict[own_number, direction, join_number])

    second_best_by_piece_direction = {
        key: secondBestScore(scores)
        for key, scores in grouped_scores.items()
    }
    for own_number, direction, join_number in list(score_dict):
        raw_score = score_dict[own_number, direction, join_number]
        second_best_score = second_best_by_piece_direction[own_number, direction]
        score_dict[own_number, direction, join_number] = reliabilityScore(
            raw_score,
            second_best_score,
        )


def secondBestScore(scores):
    sorted_scores = sorted(scores)
    if len(sorted_scores) < 2:
        return sorted_scores[0] if sorted_scores else 0.0
    return sorted_scores[1]


def reliabilityScore(score, second_best_score):
    if second_best_score <= 0:
        if score <= 0:
            return 1.0
        return float("inf")
    return score / second_best_score


def normalizeScores(segment_list, score_algorithm):
    if score_algorithm == ScoreAlgorithm.GIST_AND_EUCLIDEAN:
        score_dict = scoreDict(segment_list)
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
            color_score = score_dict[value][0]
            distance_score = score_dict[value][1]
            normalized_color_score = (color_score-min1)/(max1-min1)
            normalized_gist_score = ((distance_score-min2) / (max2-min2)
                              )  # extra weight to GIST
            score_dict[value] = normalized_color_score+normalized_gist_score
    elif score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        score_dict = scoreDict(segment_list)
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
            normalized_euclidean_score = ((euclidean_score-min2) / (max2-min2)
                              )  # extra weight to GIST
            score_dict[value] = normalized_mahalanobis_score+normalized_euclidean_score
