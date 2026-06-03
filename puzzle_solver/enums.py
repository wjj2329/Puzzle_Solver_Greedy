from enum import Enum


class JoinDirection(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


OPPOSITE_DIRECTIONS = {
    JoinDirection.UP: JoinDirection.DOWN,
    JoinDirection.DOWN: JoinDirection.UP,
    JoinDirection.LEFT: JoinDirection.RIGHT,
    JoinDirection.RIGHT: JoinDirection.LEFT,
}

JOIN_EDGE_PAIRS = [
    (JoinDirection.UP, JoinDirection.DOWN),
    (JoinDirection.DOWN, JoinDirection.UP),
    (JoinDirection.LEFT, JoinDirection.RIGHT),
    (JoinDirection.RIGHT, JoinDirection.LEFT),
]


class CompareWithOtherSegments(Enum):
    ONLY_BEST = 1
    COMPARE_WITH_SECOND = 2


class ScoreAlgorithm(Enum):
    EUCLIDEAN = 1
    MAHALANOBIS = 2
    EUCLIDEAN_AND_MAHALANOBIS = 4
    MGC = 5
    MGC_DISTANCE = 6
    PREDICTION = 7
    MGC_PREDICTION = 8


class ScoreMode(Enum):
    DISSIMILARITY = 1
    RELIABILITY = 2


class ColorType(Enum):
    RGB = 1
    LAB = 2


class AssemblyType(Enum):
    KRUSKAL = 1
    PRIM = 2
