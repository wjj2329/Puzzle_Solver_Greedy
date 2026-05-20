import argparse
import os
import numpy as np
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import copy
from enum import Enum
from pathlib import Path
import imageio.v3 as iio
from scipy.ndimage import binary_dilation
from PIL import Image
from skimage import color
from numpy import logical_and, zeros, nonzero, argwhere, delete, asarray
from numpy import sum as numpySum
from numpy import all as numpyAll
import subprocess
import scipy.signal


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


class ScoreAlgorithum(Enum):
    EUCLIDEAN = 1
    MAHALANOBIS = 2
    GIST_AND_EUCLDEAN = 3
    EUCLIDEAN_AND_MAHALANOBIS = 4


class ColorType(Enum):
    RGB = 1
    LAB = 2


class AssemblyType(Enum):
    KRUSKAL = 1
    PRIM = 2


class ScoreEdge:
    def __init__(self, edge, adjacent_edge):
        self.edge = edge
        self.adjacent_edge = adjacent_edge
        self.average_delta = np.average(edge - adjacent_edge, axis=0)
        self.inverse_covariance = np.linalg.pinv(np.cov(edge.T))


class ScorePayload:
    def __init__(self, piece_number, own_edges, compare_edges):
        self.piece_number = piece_number
        self.own_edges = own_edges
        self.compare_edges = compare_edges


_SCORE_PAYLOADS = None
_SCORE_ALGORITHM = None


def prepareImageForWrite(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and image.min() >= 0 and image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).round().astype(np.uint8)
    return image


def euclideanDistance(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff.reshape(diff.shape[0], -1)
    return float(np.linalg.norm(diff, axis=1).sum())


def mahalanobisEdgeDistance(own_edge, compare_edge):
    matrix = (own_edge.edge - compare_edge.edge) - own_edge.average_delta
    matrix2 = (compare_edge.edge - own_edge.edge) - compare_edge.average_delta

    scores = np.einsum(
        "ij,jk,ik->i", matrix, compare_edge.inverse_covariance, matrix)
    scores2 = np.einsum(
        "ij,jk,ik->i", matrix2, own_edge.inverse_covariance, matrix2)
    return float(np.sqrt(np.abs(scores)).sum() + np.sqrt(np.abs(scores2)).sum())


def reciprocalScoreEntries(own_number, join_number, scores_by_direction):
    entries = []
    for direction, score in scores_by_direction:
        entries.append(((own_number, direction, join_number), score))
        entries.append(((join_number, OPPOSITE_DIRECTIONS[direction], own_number), score))
    return entries


def scorePayloadPair(segment1, segment2, score_algorithum):
    own_edges = segment1.own_edges
    compare_edges = segment2.compare_edges
    if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
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
    elif score_algorithum == ScoreAlgorithum.MAHALANOBIS:
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
    elif score_algorithum == ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS:
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


def initializeScoreWorker(score_payloads, score_algorithum):
    global _SCORE_PAYLOADS, _SCORE_ALGORITHM
    _SCORE_PAYLOADS = score_payloads
    _SCORE_ALGORITHM = score_algorithum


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


class BestConnection:
    score = sys.maxsize
    second_best_score = sys.maxsize

    def __init__(self, own_segment=None, pic_connection_matix=None, join_segment=None, binary_connection_matrix=None):
        self.own_segment = own_segment
        self.pic_connection_matix = pic_connection_matix
        self.join_segment = join_segment
        self.binary_connection_matrix = binary_connection_matrix

    def setThings(self, pic_connection_matix, join_segment, score, own_segment, binary_connection_matrix):
        self.second_best_score = self.score
        self.pic_connection_matix = pic_connection_matix
        self.join_segment = join_segment
        self.score = score
        self.own_segment = own_segment
        self.binary_connection_matrix = binary_connection_matrix

    def isBetterConnection(self, otherConnection, compare_type):
        if compare_type == CompareWithOtherSegments.ONLY_BEST:
            return self.score < otherConnection.score
        elif compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
            return (2*(self.score*((self.score/self.second_best_score))))+self.score < otherConnection.score+(2*(otherConnection.score*((otherConnection.score/otherConnection.second_best_score))))

    def stripZeros(self):
        self.pic_connection_matix = self.pic_connection_matix[~numpyAll(
            self.pic_connection_matix == 0, axis=1)]
        self.binary_connection_matrix = self.binary_connection_matrix[~numpyAll(
            self.binary_connection_matrix == 0, axis=1)]
        idx = argwhere(
            numpyAll(self.pic_connection_matix[..., :] == 0, axis=0))
        self.pic_connection_matix = delete(
            self.pic_connection_matix, idx, axis=1)
        idx = argwhere(
            numpyAll(self.binary_connection_matrix[..., :] == 0, axis=0))
        self.binary_connection_matrix = delete(
            self.binary_connection_matrix, idx, axis=1)

    def __eq__(self, other):
        return self.score == other.score


class Segment:
    dilation_mask = asarray([[0, 1, 0], [1, 1, 1, ], [0, 1, 0]])
    binary_connection_matrix = asarray([[1, 0], [0, 0]])
    best_connection_found_so_far = BestConnection()

    def __init__(self, pic_matrix, max_width, max_height, piece_number, myownNumber, score_dict, gist, connections_dict):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = asarray([[self, 0], [0, 0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number
        self.myownNumber = myownNumber
        self.score_dict = score_dict
        self.gist = gist
        self.connections_dict = connections_dict
        self._own_score_edges = None
        self._compare_score_edges = None

    def __add__(self, other):
        if type(self) is Segment:
            return self
        else:
            return other

    def __radd__(self, other):
        if type(self) is Segment:
            return self
        else:
            return other

    def euclideanDistance(self, a, b):
        return euclideanDistance(a, b)

    def buildScoreEdges(self, pic_matrix):
        return {
            JoinDirection.UP: ScoreEdge(pic_matrix[0, :, :], pic_matrix[1, :, :]),
            JoinDirection.DOWN: ScoreEdge(pic_matrix[-1, :, :], pic_matrix[-2, :, :]),
            JoinDirection.LEFT: ScoreEdge(pic_matrix[:, 0, :], pic_matrix[:, 1, :]),
            JoinDirection.RIGHT: ScoreEdge(pic_matrix[:, -1, :], pic_matrix[:, -2, :]),
        }

    def ownScoreEdges(self):
        if self._own_score_edges is None:
            self._own_score_edges = self.buildScoreEdges(self.pic_matrix.astype(np.int16))
        return self._own_score_edges

    def compareScoreEdges(self):
        if self._compare_score_edges is None:
            self._compare_score_edges = self.buildScoreEdges(self.pic_matrix)
        return self._compare_score_edges

    def reciprocalScoreEntries(self, segment, scores_by_direction):
        return reciprocalScoreEntries(
            self.piece_number,
            segment.piece_number,
            scores_by_direction,
        )

    def scoreEntriesEuclidean(self, segment):
        own_edges = self.ownScoreEdges()
        compare_edges = segment.compareScoreEdges()
        return self.reciprocalScoreEntries(
            segment,
            [
                (
                    own_direction,
                    self.euclideanDistance(
                        own_edges[own_direction].edge,
                        compare_edges[compare_direction].edge,
                    ),
                )
                for own_direction, compare_direction in JOIN_EDGE_PAIRS
            ],
        )

    def scoreEntriesMahalonbis(self, segment):
        own_edges = self.ownScoreEdges()
        compare_edges = segment.compareScoreEdges()
        return self.reciprocalScoreEntries(
            segment,
            [
                (
                    own_direction,
                    self.mahalanobisEdgeDistance(
                        own_edges[own_direction],
                        compare_edges[compare_direction],
                    ),
                )
                for own_direction, compare_direction in JOIN_EDGE_PAIRS
            ],
        )

    def scoreEntriesEuclideanAndMahalonbis(self, segment):
        own_edges = self.ownScoreEdges()
        compare_edges = segment.compareScoreEdges()
        return self.reciprocalScoreEntries(
            segment,
            [
                (
                    own_direction,
                    (
                        self.mahalanobisEdgeDistance(
                            own_edges[own_direction],
                            compare_edges[compare_direction],
                        ),
                        self.euclideanDistance(
                            own_edges[own_direction].edge,
                            compare_edges[compare_direction].edge,
                        ),
                    ),
                )
                for own_direction, compare_direction in JOIN_EDGE_PAIRS
            ],
        )

    def applyScoreEntries(self, entries):
        for key, score in entries:
            self.score_dict[key] = score

    def gistDistance(self, a, b, segment):
        colorScore = self.euclideanDistance(a, b)
        gistScore = self.euclideanDistance(
            np.asarray([self.gist]), np.asarray([segment.gist]))
        return (colorScore, gistScore)

    def mahalanobisDistance(self, a, a2, z, z2):
        return self.mahalanobisEdgeDistance(ScoreEdge(a, a2), ScoreEdge(z, z2))

    def mahalanobisEdgeDistance(self, own_edge, compare_edge):
        return mahalanobisEdgeDistance(own_edge, compare_edge)

    def calculateScoreMahalonbis(self, segment):
        self.applyScoreEntries(self.scoreEntriesMahalonbis(segment))

    def calculateScoreGIST(self, segment):  # this doesn't work :(
        size = segment.pic_matrix.shape[0]
        score_dict = self.score_dict
        gistDistance = self.gistDistance

        pic_matrix = self.pic_matrix
        self_top = pic_matrix[0:1, :, :]
        self_left = np.rot90(pic_matrix[:, 0:1, :])
        self_bottom = pic_matrix[size - 1:size, :, :]
        self_right = np.rot90(pic_matrix[:, size - 1:size, :])

        segment_matrix = segment.pic_matrix
        compare_top = segment_matrix[0:1, :, :]
        compare_left = np.rot90(segment_matrix[:, 0:1, :])
        compare_bottom = segment_matrix[size - 1:size, :, :]
        compare_right = np.rot90(segment_matrix[:, size - 1:size, :])

        own_number = self.piece_number
        join_number = segment.piece_number
        score_dict[own_number, JoinDirection.UP,
                   join_number] = gistDistance(self_top, compare_bottom, segment)
        score_dict[own_number, JoinDirection.DOWN,
                   join_number] = gistDistance(self_bottom, compare_top, segment)
        score_dict[own_number, JoinDirection.LEFT,
                   join_number] = gistDistance(self_left, compare_right, segment)
        score_dict[own_number, JoinDirection.RIGHT,
                   join_number] = gistDistance(self_right, compare_left, segment)

        score_dict[join_number, JoinDirection.DOWN,
                   own_number] = score_dict[own_number, JoinDirection.UP,
                                            join_number]
        score_dict[join_number, JoinDirection.UP,
                   own_number] = score_dict[own_number, JoinDirection.DOWN,
                                            join_number]
        score_dict[join_number, JoinDirection.RIGHT,
                   own_number] = score_dict[own_number, JoinDirection.LEFT,
                                            join_number]
        score_dict[join_number, JoinDirection.LEFT,
                   own_number] = score_dict[own_number, JoinDirection.RIGHT,
                                            join_number]

    def calculateScoreEuclidean(self, segment):
        self.applyScoreEntries(self.scoreEntriesEuclidean(segment))

    def calculateScoreEuclideanAndMahalonbis(self, segment):
        self.applyScoreEntries(self.scoreEntriesEuclideanAndMahalonbis(segment))

    def checkforcompatibility(self, booleanarray, max_height, max_width):
        non_zero_values = nonzero(booleanarray)
        smallestx1 = min(non_zero_values[1])
        smallesty1 = min(non_zero_values[0])
        biggestx1 = max(non_zero_values[1])
        biggesty1 = max(non_zero_values[0])
        if biggestx1-smallestx1 > max_height or biggesty1-smallesty1 > max_width:
            return False
        return True

    def calculateConnectionsPrim(self, compare_segment):
        best_connection_found_so_far = self.best_connection_found_so_far
        shape = self.binary_connection_matrix.shape
        self_binary_matrix = np.zeros((shape[0]+4, shape[1]+4))
        self_binary_matrix[2:shape[0]+2, 2:shape[1] +
                           2] = self.binary_connection_matrix
        self_pic_matrix = np.zeros((shape[0]+4, shape[1]+4), dtype="object")
        self_pic_matrix[2:shape[0]+2, 2:shape[1]+2] = self.pic_connection_matix
        pieces_to_check = self_pic_matrix.nonzero()
        score_dict = self.score_dict
        checkforcompatibility = self.checkforcompatibility
        compare_segment_piece_number = compare_segment.piece_number
        for x, y in zip(pieces_to_check[0], pieces_to_check[1]):
            if self_pic_matrix[x+1][y] == 0:
                score = 0
                numberofsides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.DOWN, compare_segment_piece_number]
                if self_pic_matrix[x+2][y] != 0:  # check piece to right down and left
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+2][y].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x+1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x+1][y+1].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x+1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x+1][y-1].piece_number]
                    numberofsides += 1
                score = score/numberofsides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x+1, y] = compare_segment
                    temp_binary_matrix[x+1, y] = 1
                    if checkforcompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setThings(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)

            if self_pic_matrix[x-1][y] == 0:
                score = 0
                numberofsides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.UP, compare_segment_piece_number]
                if self_pic_matrix[x-2][y] != 0:  # check piece to right down and left
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-2][y].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x-1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x-1][y+1].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x-1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x-1][y-1].piece_number]
                    numberofsides += 1
                score = score/numberofsides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x-1, y] = compare_segment
                    temp_binary_matrix[x-1, y] = 1
                    if checkforcompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setThings(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
            if self_pic_matrix[x][y+1] == 0:
                score = 0
                numberofsides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.RIGHT, compare_segment_piece_number]
                if self_pic_matrix[x][y+2] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x][y+2].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x+1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+1][y+1].piece_number]
                    numberofsides += 1

                if self_pic_matrix[x-1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-1][y+1].piece_number]
                    numberofsides += 1
                score = score/numberofsides

                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x, y+1] = compare_segment
                    temp_binary_matrix[x, y+1] = 1
                    if checkforcompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setThings(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
            if self_pic_matrix[x][y-1] == 0:
                score = 0
                numberofsides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.LEFT, compare_segment_piece_number]
                if self_pic_matrix[x][y-2] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x][y-2].piece_number]
                    numberofsides += 1
                if self_pic_matrix[x+1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+1][y-1].piece_number]
                    numberofsides += 1
                if self_pic_matrix[x-1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-1][y-1].piece_number]
                    numberofsides += 1
                score = score/numberofsides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x, y-1] = compare_segment
                    temp_binary_matrix[x, y-1] = 1
                    if checkforcompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setThings(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
        return best_connection_found_so_far

    # sadly this is slower than just brute forcing the entire thing
    def findValuesToCompare(self, a):
        p_a = np.pad(a, 1, mode='constant', constant_values=1)
        window = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        m = scipy.signal.convolve2d(p_a, window, mode='same')
        v = np.where((a == 1) & (m[1:-1, 1:-1] < 4))
        return v

    def calculateConnectionsKruskal(self, compare_segment, boost_priority_of_big_pieces_joining):
        if (self.myownNumber, compare_segment.myownNumber) in self.connections_dict:
            return self.connections_dict[(self.myownNumber, compare_segment.myownNumber)]
        checkforcompatibility = self.checkforcompatibility
        score_dict = self.score_dict
        best_connection_found_so_far = self.best_connection_found_so_far
        own_binary_connection_matrix = self.binary_connection_matrix
        compare_segment_binary_connection_matrix = compare_segment.binary_connection_matrix
        own_pic_connection_matrix = self.pic_connection_matix
        compare_segment_pic_connection_matrix = compare_segment.pic_connection_matix
        h1 = own_binary_connection_matrix.shape[0]
        w1 = own_binary_connection_matrix.shape[1]
        h2 = compare_segment_binary_connection_matrix.shape[0]
        w2 = compare_segment_binary_connection_matrix.shape[1]
        height_padded = h1+2*h2
        width_padded = w1+2*w2
        height_combined = h2+h1
        width_combined = w2+w1
        max_height = self.max_height
        max_width = self.max_width
        pad_with_piece1 = zeros((height_padded, width_padded))
        pad_with_piece1[h2:height_combined, w2:(
            width_combined)] = own_binary_connection_matrix
        neighboring_connections = binary_dilation(
            input=pad_with_piece1, structure=self.dilation_mask) - pad_with_piece1
        neighboring_connections_shape = neighboring_connections.shape
        padded1_pointer = zeros(
            (height_padded, width_padded), dtype="object")
        padded1_pointer[h2:(height_combined), w2:(
                        width_combined)] = own_pic_connection_matrix
        for x in range(height_padded-(h2-1)):
            for y in range(width_padded-(w2-1)):
                pad_with_piece2 = zeros(neighboring_connections_shape)
                pad_with_piece2[x:(x+h2), y:(y+w2)
                                ] = compare_segment_binary_connection_matrix
                if not numpySum(logical_and(
                        neighboring_connections, pad_with_piece2)[:]) > 0:
                    continue
                if numpySum(logical_and(
                        pad_with_piece1, pad_with_piece2)[:]) > 0:
                    continue
                combined_pieces = pad_with_piece1+pad_with_piece2
                if checkforcompatibility(combined_pieces, max_height, max_width):
                    store = nonzero(pad_with_piece1)
                    score = 0
                    numofcompar = 0
                    temp_pointer = zeros(  
                        (height_padded, width_padded), dtype="object")

                    temp_pointer[x:(h2+x), y:(w2+y)
                                 ] = compare_segment_pic_connection_matrix
                    combined_pointer = temp_pointer+padded1_pointer
                    for d, h in zip(store[0], store[1]):
                        node1 = combined_pointer[d, h].piece_number
                        if pad_with_piece2[d][h+1] == 1:
                            numofcompar += 1
                            score += score_dict[node1,
                                                JoinDirection.RIGHT, combined_pointer[d, h+1].piece_number]
                        if pad_with_piece2[d][h-1] == 1:
                            numofcompar += 1
                            score += score_dict[node1,
                                                JoinDirection.LEFT, combined_pointer[d, h-1].piece_number]
                        if pad_with_piece2[d+1][h] == 1:
                            numofcompar += 1
                            score += score_dict[node1,
                                                JoinDirection.DOWN, combined_pointer[d+1, h].piece_number]
                        if pad_with_piece2[d-1][h] == 1:
                            numofcompar += 1
                            score += score_dict[node1,
                                                JoinDirection.UP, combined_pointer[d-1, h].piece_number]
                    if boost_priority_of_big_pieces_joining:
                        score = score/((numofcompar*numofcompar)*0.5)
                    else:
                        score = score/numofcompar
                    if score < best_connection_found_so_far.score:
                        best_connection_found_so_far.setThings(
                            combined_pointer, compare_segment, score, self, combined_pieces)
        self.connections_dict[(
            self.myownNumber, compare_segment.myownNumber)] = best_connection_found_so_far
        return best_connection_found_so_far


def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputpic", action="store",
                        help="add picture you want to run on", required=True)
    parser.add_argument("-sp", "--savepieces", action="store_true",
                        help="save the pieces the picture was broken up into", default=False)
    parser.add_argument("-l", "--length", action="store", type=int,
                        help="size of the length of square segments wanted in pixels", required=True)
    parser.add_argument("-sa", "--saveassembly", action="store_true",
                        help="save the assembled picture in each round", default=False)
    parser.add_argument("-a", "--showanimation", action="store_true",
                        help="show animation of picture being built", default=True)
    parser.add_argument("-k", "--use kruskal for building",
                        action="store_true")
    parser.add_argument("-p", "-use prims for building", action="store_true")
    return parser.parse_args()


def get_gist(filename):
    data = open(filename, 'r').read()
    return [float(x) for x in data.split()]


def breakUpImage(image, length, save_segments, colortype, score_algorithum):
    dimensions = image.shape
    if dimensions[0] != dimensions[1]:
        print("Only square images will work for now to keep things simple")
        exit()
    if dimensions[0] % length != 0 or dimensions[1] % length != 0:
        print("unable to break up image into equal squares")
        exit()
    segments = []
    x, y = 0, 0
    picX, picY = 0, 0
    piece_num = 1
    num_of_pieces_width = int(dimensions[0]/length)
    num_of_pieces_height = int(dimensions[1]/length)
    append = segments.append
    score_dict = {}
    connections_dict = {}
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[picX: picX+length, picY: picY+length, :]
            gist = None
            if save_segments:
                if colortype == ColorType.RGB:
                    iio.imwrite(str(x)+"_"+str(y)+".png", prepareImageForWrite(save))
                elif colortype == ColorType.LAB:
                    imageTemp = color.lab2rgb(save)
                    iio.imwrite(str(x)+"_"+str(y)+".png", prepareImageForWrite(imageTemp))
                elif score_algorithum == ScoreAlgorithum.GIST_AND_EUCLDEAN:
                    subprocess.run(["gist.exe", "-i", "C:\\Users\\wjones\\Desktop\\puzzle_solver\\Puzzle_Solver_Greedy\\Python3\\"+str(
                        x)+"_"+str(y)+".png", "-o", "C:\\Users\\wjones\\Desktop\\puzzle_solver\\Puzzle_Solver_Greedy\\Python3"])
                    gist = get_gist("gist.txt")
            segment_to_append = Segment(save, num_of_pieces_width,
                                        num_of_pieces_height, piece_num, piece_num, score_dict, gist, connections_dict)
            append(segment_to_append)
            piece_num += 1
            picY += length
        picX += length
        picY = 0
    return segments


def scoreEntriesForPair(segment1, segment2, score_algorithum):
    if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
        return segment1.scoreEntriesEuclidean(segment2)
    elif score_algorithum == ScoreAlgorithum.MAHALANOBIS:
        return segment1.scoreEntriesMahalonbis(segment2)
    elif score_algorithum == ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS:
        return segment1.scoreEntriesEuclideanAndMahalonbis(segment2)
    return None


def scoreEntriesForSegment(segment1, remaining_segments, score_algorithum):
    entries = []
    for segment2 in remaining_segments:
        entries.extend(scoreEntriesForPair(segment1, segment2, score_algorithum))
    return entries


def calculateScoresSerial(segment_list, score_algorithum, show_progress=True):
    for index, segment1 in enumerate(segment_list):
        if show_progress:
            print("calculating score for segment ", segment1.piece_number)
        for segment2 in segment_list[index+1:]:
            if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
                segment1.calculateScoreEuclidean(segment2)
            elif score_algorithum == ScoreAlgorithum.MAHALANOBIS:
                segment1.calculateScoreMahalonbis(segment2)
            elif score_algorithum == ScoreAlgorithum.GIST_AND_EUCLDEAN:
                segment1.calculateScoreGIST(segment2)
            elif score_algorithum == ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS:
                segment1.calculateScoreEuclideanAndMahalonbis(segment2)


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


def calculateScoresThreaded(segment_list, score_algorithum, show_progress=True, max_workers=None):
    if len(segment_list) < 2:
        return
    if max_workers is None:
        max_workers = min(len(segment_list), os.cpu_count() or 1)
    if max_workers <= 1:
        calculateScoresSerial(segment_list, score_algorithum, show_progress)
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
                score_algorithum,
            ))
        for future in as_completed(futures):
            for key, score in future.result():
                score_dict[key] = score


def calculateScoresProcess(segment_list, score_algorithum, show_progress=True, max_workers=None):
    if len(segment_list) < 2:
        return
    if max_workers is None:
        max_workers = min(len(segment_list), os.cpu_count() or 1)
    if max_workers <= 1:
        calculateScoresSerial(segment_list, score_algorithum, show_progress)
        return

    score_payloads = buildScorePayloads(segment_list)
    score_dict = segment_list[0].score_dict
    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializeScoreWorker,
            initargs=(score_payloads, score_algorithum)) as executor:
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


def calculateScores(segment_list, score_algorithum, show_progress=True, max_workers=None, executor_type="thread"):
    if score_algorithum == ScoreAlgorithum.GIST_AND_EUCLDEAN:
        calculateScoresSerial(segment_list, score_algorithum, show_progress)
    elif executor_type == "serial":
        calculateScoresSerial(segment_list, score_algorithum, show_progress)
    elif executor_type == "process":
        calculateScoresProcess(segment_list, score_algorithum, show_progress, max_workers)
    elif executor_type == "thread":
        calculateScoresThreaded(segment_list, score_algorithum, show_progress, max_workers)
    else:
        raise ValueError("executor_type must be 'serial', 'thread', or 'process'")


def findBestConnectionKruskal(segment_list, compare_type, boost_priority_of_big_pieces_joining, compareType):
    best_so_far = BestConnection()
    if compareType == CompareWithOtherSegments.ONLY_BEST:
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


def findBestConnectionPrim(segment_list, rootSegment, compare_type):
    best_so_far = BestConnection()
    for segment in segment_list:
        if segment != rootSegment:
            rootSegment.best_connection_found_so_far = BestConnection()
            temp = rootSegment.calculateConnectionsPrim(segment)
            if temp.isBetterConnection(best_so_far, compare_type):
                best_so_far = temp
    return best_so_far


# TODO no idea how to find the best one to start,  will do random for now! Maybe find piece with best connections
def findBestRootSegment(segment_list):
    return random.choice(segment_list)


def printPiecesMatrices(segment_list):
    for node in segment_list:
        print(node.binary_connection_matrix)
        print(node.pic_connection_matix)
        print(node.piece_number)
    print('\n\n\n')


def clearDictionaryForRam(my_list, removing):
    for connection in my_list:
        for key in dict(connection.connections_dict):
            if key[0] == removing or key[1] == removing:
                del connection.connections_dict[key]


def saveImage(best_connection, piece_size, round, colortype, name_for_round):
    pic_locations = best_connection.binary_connection_matrix.nonzero()
    biggestx = max(pic_locations[0])
    biggesty = max(pic_locations[1])
    smallestx = min(pic_locations[0])
    smallesty = min(pic_locations[1])
    sizex = (biggestx-smallestx)+1
    sizey = (biggesty-smallesty)+1
    biggest_dim = sizex if sizex > sizey else sizey
    new_image = zeros((biggest_dim*piece_size, biggest_dim*piece_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matix[pic_locations[0]
                                                                 [x], pic_locations[1][x]].pic_matrix
        x1 = (pic_locations[0][x]-smallestx)*piece_size
        y1 = (pic_locations[1][x]-smallesty)*piece_size
        new_image[x1:x1+piece_size, y1:y1+piece_size, :] = piece_to_assemble
    if colortype == ColorType.LAB:
        new_image = color.lab2rgb(new_image)
    imageName = name_for_round+" round"+str(round)+".png"
    iio.imwrite(imageName, prepareImageForWrite(new_image))
    return imageName


def normalizeScores(segment_list, scoreType):
    if scoreType == ScoreAlgorithum.GIST_AND_EUCLDEAN:
        score_dict = segment_list[0].score_dict
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
            colorScore = score_dict[value][0]
            distScore = score_dict[value][1]
            colorScoreNormal = (colorScore-min1)/(max1-min1)
            colorScoreGIST = ((distScore-min2) / (max2-min2)
                              )  # extra weight to GIST
            score_dict[value] = colorScoreNormal+colorScoreGIST
    elif scoreType == ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS:
        score_dict = segment_list[0].score_dict
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
            colorScore = score_dict[value][0]
            distScore = score_dict[value][1]
            colorScoreMal = (colorScore-min1)/(max1-min1)
            colorScoreEucl = ((distScore-min2) / (max2-min2)
                              )  # extra weight to GIST
            score_dict[value] = colorScoreMal+colorScoreEucl            


def createCrossPiece(segment_list):
    root = findBestRootSegment(segment_list)


def checkFunctionCacsTheSameOnEachPeice(segment_list, boost_priority_of_big_pieces_joining):
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
    best_so_far = BestConnection()
    for segment2 in segment_list:
        if segment != segment2:
            segment.best_connection_found_so_far = BestConnection()
            temp = segment.calculateConnectionsKruskal(segment2, False)
            if temp.isBetterConnection(best_so_far, CompareWithOtherSegments.ONLY_BEST):
                best_so_far = temp
    return best_so_far


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
    best_connection.own_segment.pic_connection_matix = best_connection.pic_connection_matix
    best_connection.own_segment.myownNumber += original_size
    segment_list.remove(best_connection.join_segment)


# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?
# TODO Maybe have it go in lines? Or at least start off with two lines one horizontal one vertical to build off and stop going out of bounds?
# TODO maybe combo of kruskal and prims? Divide into blocks? Limit the number of trees? Force to use prims after awhile?
# TODO do a best budy where each peice thinks the other is the best and get those done FIRST
# TODO Different color spaces
# TODO Find balance of second best ratio
# TODO is mal distance the same either way???? Did I get that wrong?
# TODO combo of Euclidean and MAL?
# http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
# https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
# https://www.python.org/dev/peps/pep-0371/ use this to make it faster
# https://www.sciencedirect.com/science/article/pii/S131915781830394X gist combo with euclidean
# https://pdfs.semanticscholar.org/4003/7d131e3365feb9d69912b3c8e8527e9ed2d5.pdf  cycle detection
# Filter the image?  Gausian blur etc?
def main():
    start_time = time.time()
    picture_file_name = Path(__file__).resolve().with_name("William.png")
    length = 30
    save_segments = True
    image = iio.imread(picture_file_name)
    save_assembly_to_disk = True
    show_building_animation = True
    show_print_statements = True
    boost_priority_of_big_pieces_joining = False
    connect_best_friends_first = True
    score_workers = None
    score_executor = "process"

    colorType = ColorType.LAB
    assemblyType = AssemblyType.KRUSKAL
    scoreType = ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS
    compareType = CompareWithOtherSegments.ONLY_BEST
    name_for_round = "test"

    if colorType == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(
        image, length, save_segments, colorType, scoreType)
    calculateScores(
        segment_list, scoreType, show_print_statements, score_workers, score_executor)

    normalizeScores(segment_list, scoreType)
    elapsed_time_secs = time.time() - start_time
    if show_print_statements:
        print("Calculate scores took: %s secs " % elapsed_time_secs)
    window, w = None, None
    if show_building_animation:
        import tkinter
        from PIL import ImageTk

        window = tkinter.Tk()
        window.title("Picture")
        img = ImageTk.PhotoImage(Image.open(picture_file_name))
        w = tkinter.Label(window, image=img)
    random.shuffle(segment_list)
    round = 0
    original_size = len(segment_list)
    root = None
    if connect_best_friends_first:
        connectBestBudsFirst(segment_list, original_size, show_print_statements)
    if assemblyType == AssemblyType.PRIM:
        root = findBestRootSegment(segment_list)
    while len(segment_list) > 1:
        best_connection = None
        if assemblyType == AssemblyType.KRUSKAL:
            best_connection = findBestConnectionKruskal(
                segment_list, compareType, boost_priority_of_big_pieces_joining, compareType)
        if assemblyType == AssemblyType.PRIM:
            best_connection = findBestConnectionPrim(
                segment_list, root, compareType)
        joinPieces(best_connection, segment_list, original_size)
        root = best_connection.own_segment
        if save_assembly_to_disk:
            image_name = saveImage(best_connection, length, round, colorType, name_for_round)
            if show_building_animation:
                updated_picture = ImageTk.PhotoImage(Image.open(image_name))
                w.configure(image=updated_picture)
                w.image = updated_picture
                w.pack(side="bottom", fill="both", expand="no")
                window.update()
        if show_print_statements == True:
            print("for round ", round, " i get score of ", best_connection.score, "the ratio for first to second best is ",
                  best_connection.score/best_connection.second_best_score, " it took ", time.time()-start_time)
        round += 1

    if show_print_statements == True:
        elapsed_time_secs = time.time() - start_time
        print("Execution took: %s secs " % elapsed_time_secs)


if __name__ == '__main__':
    main()
