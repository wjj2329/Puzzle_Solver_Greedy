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
    GIST_AND_EUCLIDEAN = 3
    EUCLIDEAN_AND_MAHALANOBIS = 4


class ColorType(Enum):
    RGB = 1
    LAB = 2


class AssemblyType(Enum):
    KRUSKAL = 1
    PRIM = 2
