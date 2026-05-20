from .distances import euclideanDistance, mahalanobisEdgeDistance
from .enums import JOIN_EDGE_PAIRS, OPPOSITE_DIRECTIONS, ScoreAlgorithm


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
    else:
        return None
    return reciprocalScoreEntries(segment1.piece_number, segment2.piece_number, scores)
