from .distances import euclideanDistance, mahalanobisEdgeDistance, mgcEdgeDistance
from .enums import (
    JOIN_EDGE_PAIRS,
    OPPOSITE_DIRECTIONS,
    JoinDirection,
    ScoreAlgorithm,
)


DIRECTION_TO_INDEX = {
    direction: index
    for index, direction in enumerate(JoinDirection)
}
JOIN_EDGE_PAIR_INDICES = tuple(
    (
        own_direction,
        compare_direction,
        DIRECTION_TO_INDEX[own_direction],
        DIRECTION_TO_INDEX[OPPOSITE_DIRECTIONS[own_direction]],
    )
    for own_direction, compare_direction in JOIN_EDGE_PAIRS
)


def reciprocalScoreEntries(own_number, join_number, scores_by_direction):
    entries = []
    for direction, score in scores_by_direction:
        entries.append(((own_number, direction, join_number), score))
        entries.append(((join_number, OPPOSITE_DIRECTIONS[direction], own_number), score))
    return entries


def scorePayloadPair(segment1, segment2, score_algorithm):
    own_edges = segment1.own_edges
    compare_edges = segment2.compare_edges
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN:
        scores = [
            (
                own_direction,
                euclideanDistance(
                    own_edges[own_direction].edge,
                    compare_edges[compare_direction].edge,
                ),
            )
            for own_direction, compare_direction in JOIN_EDGE_PAIRS
        ]
    elif score_algorithm == ScoreAlgorithm.MAHALANOBIS:
        scores = [
            (
                own_direction,
                mahalanobisEdgeDistance(
                    own_edges[own_direction],
                    compare_edges[compare_direction],
                ),
            )
            for own_direction, compare_direction in JOIN_EDGE_PAIRS
        ]
    elif score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        scores = [
            (
                own_direction,
                (
                    mahalanobisEdgeDistance(
                        own_edges[own_direction],
                        compare_edges[compare_direction],
                    ),
                    euclideanDistance(
                        own_edges[own_direction].edge,
                        compare_edges[compare_direction].edge,
                    ),
                ),
            )
            for own_direction, compare_direction in JOIN_EDGE_PAIRS
        ]
    elif score_algorithm == ScoreAlgorithm.MGC:
        scores = [
            (
                own_direction,
                mgcEdgeDistance(
                    own_edges[own_direction],
                    compare_edges[compare_direction],
                ),
            )
            for own_direction, compare_direction in JOIN_EDGE_PAIRS
        ]
    else:
        return None
    return reciprocalScoreEntries(segment1.piece_number, segment2.piece_number, scores)


def scoreComponentCount(score_algorithm):
    if score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        return 2
    return 1


def appendScorePayloadPairArrayValues(
        segment1,
        segment2,
        score_algorithm,
        own_numbers,
        direction_indices,
        join_numbers,
        scores,
        cursor):
    own_edges = segment1.own_edges
    compare_edges = segment2.compare_edges
    own_number = segment1.piece_number
    join_number = segment2.piece_number
    own_numbers_append = own_numbers.append
    direction_indices_append = direction_indices.append
    join_numbers_append = join_numbers.append

    if score_algorithm == ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS:
        mahalanobis_scores, euclidean_scores = scores
        mahalanobis_scores_append = mahalanobis_scores.append
        euclidean_scores_append = euclidean_scores.append
        for (
                own_direction,
                compare_direction,
                direction_index,
                opposite_direction_index) in JOIN_EDGE_PAIR_INDICES:
            mahalanobis_score = mahalanobisEdgeDistance(
                own_edges[own_direction],
                compare_edges[compare_direction],
            )
            euclidean_score = euclideanDistance(
                own_edges[own_direction].edge,
                compare_edges[compare_direction].edge,
            )
            own_numbers_append(own_number)
            direction_indices_append(direction_index)
            join_numbers_append(join_number)
            mahalanobis_scores_append(mahalanobis_score)
            euclidean_scores_append(euclidean_score)
            cursor += 1

            own_numbers_append(join_number)
            direction_indices_append(opposite_direction_index)
            join_numbers_append(own_number)
            mahalanobis_scores_append(mahalanobis_score)
            euclidean_scores_append(euclidean_score)
            cursor += 1
        return cursor

    scores_append = scores.append
    for (
            own_direction,
            compare_direction,
            direction_index,
            opposite_direction_index) in JOIN_EDGE_PAIR_INDICES:
        if score_algorithm == ScoreAlgorithm.EUCLIDEAN:
            score = euclideanDistance(
                own_edges[own_direction].edge,
                compare_edges[compare_direction].edge,
            )
        elif score_algorithm == ScoreAlgorithm.MAHALANOBIS:
            score = mahalanobisEdgeDistance(
                own_edges[own_direction],
                compare_edges[compare_direction],
            )
        elif score_algorithm == ScoreAlgorithm.MGC:
            score = mgcEdgeDistance(
                own_edges[own_direction],
                compare_edges[compare_direction],
            )
        else:
            return cursor

        own_numbers_append(own_number)
        direction_indices_append(direction_index)
        join_numbers_append(join_number)
        scores_append(score)
        cursor += 1

        own_numbers_append(join_number)
        direction_indices_append(opposite_direction_index)
        join_numbers_append(own_number)
        scores_append(score)
        cursor += 1
    return cursor
