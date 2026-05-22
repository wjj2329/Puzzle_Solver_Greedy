import sys
from copy import copy

import numpy as np
from numpy import argwhere, asarray, delete, nonzero, zeros
from numpy import all as numpy_all

from .distances import (
    MGC_DUMMY_GRADIENTS,
    euclideanDistance,
    mahalanobisEdgeDistance,
    mgcEdgeDistance,
)
from .enums import CompareWithOtherSegments, JOIN_EDGE_PAIRS, JoinDirection
from .score_helpers import reciprocalScoreEntries


class ScoreEdge:
    def __init__(self, edge, adjacent_edge):
        self.edge = np.asarray(edge, dtype=np.float64)
        self.adjacent_edge = np.asarray(adjacent_edge, dtype=np.float64)
        self.gradient = self.edge - self.adjacent_edge
        self.average_delta = np.average(self.gradient, axis=0)
        self.gradient_average = self.average_delta
        self.inverse_covariance = np.linalg.pinv(np.cov(self.edge.T))
        gradient_samples = np.vstack((self.gradient, MGC_DUMMY_GRADIENTS))
        self.gradient_inverse_covariance = np.linalg.pinv(
            np.cov(gradient_samples.T))


class ScorePayload:
    def __init__(self, piece_number, own_edges, compare_edges):
        self.piece_number = piece_number
        self.own_edges = own_edges
        self.compare_edges = compare_edges


class BestConnection:
    score = sys.maxsize
    second_best_score = sys.maxsize

    def __init__(self, own_segment=None, pic_connection_matrix=None, join_segment=None, binary_connection_matrix=None):
        self.own_segment = own_segment
        self.pic_connection_matrix = pic_connection_matrix
        self.join_segment = join_segment
        self.binary_connection_matrix = binary_connection_matrix

    def setConnection(self, pic_connection_matrix, join_segment, score, own_segment, binary_connection_matrix):
        self.second_best_score = self.score
        self.pic_connection_matrix = pic_connection_matrix
        self.join_segment = join_segment
        self.score = score
        self.own_segment = own_segment
        self.binary_connection_matrix = binary_connection_matrix

    def isBetterConnection(self, other_connection, compare_type):
        if compare_type == CompareWithOtherSegments.ONLY_BEST:
            return self.score < other_connection.score
        elif compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
            return (2*(self.score*((self.score/self.second_best_score))))+self.score < other_connection.score+(2*(other_connection.score*((other_connection.score/other_connection.second_best_score))))

    def stripZeros(self):
        self.pic_connection_matrix = self.pic_connection_matrix[~numpy_all(
            self.pic_connection_matrix == 0, axis=1)]
        self.binary_connection_matrix = self.binary_connection_matrix[~numpy_all(
            self.binary_connection_matrix == 0, axis=1)]
        idx = argwhere(
            numpy_all(self.pic_connection_matrix[..., :] == 0, axis=0))
        self.pic_connection_matrix = delete(
            self.pic_connection_matrix, idx, axis=1)
        idx = argwhere(
            numpy_all(self.binary_connection_matrix[..., :] == 0, axis=0))
        self.binary_connection_matrix = delete(
            self.binary_connection_matrix, idx, axis=1)

    def __eq__(self, other):
        return self.score == other.score


class Segment:
    dilation_mask = asarray([[0, 1, 0], [1, 1, 1, ], [0, 1, 0]])
    binary_connection_matrix = asarray([[1, 0], [0, 0]])
    best_connection_found_so_far = BestConnection()

    def __init__(self, pic_matrix, max_width, max_height, piece_number, component_id, score_dict, connections_dict):
        self.pic_matrix = pic_matrix
        self.pic_connection_matrix = asarray([[self, 0], [0, 0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number
        self.component_id = component_id
        self.score_dict = score_dict
        self.connections_dict = connections_dict
        self._own_score_edges = None
        self._compare_score_edges = None
        self._kruskal_component_data = None

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
            self._own_score_edges = self.buildScoreEdges(self.pic_matrix)
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

    def scoreEntriesMahalanobis(self, segment):
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

    def scoreEntriesEuclideanAndMahalanobis(self, segment):
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

    def scoreEntriesMGC(self, segment):
        own_edges = self.ownScoreEdges()
        compare_edges = segment.compareScoreEdges()
        return self.reciprocalScoreEntries(
            segment,
            [
                (
                    own_direction,
                    self.mgcEdgeDistance(
                        own_edges[own_direction],
                        compare_edges[compare_direction],
                    ),
                )
                for own_direction, compare_direction in JOIN_EDGE_PAIRS
            ],
        )

    def applyScoreEntries(self, entries):
        for key, score in entries:
            self.score_dict[key] = score

    def mahalanobisDistance(self, a, a2, z, z2):
        return self.mahalanobisEdgeDistance(ScoreEdge(a, a2), ScoreEdge(z, z2))

    def mahalanobisEdgeDistance(self, own_edge, compare_edge):
        return mahalanobisEdgeDistance(own_edge, compare_edge)

    def mgcEdgeDistance(self, own_edge, compare_edge):
        return mgcEdgeDistance(own_edge, compare_edge)

    def calculateScoreMahalanobis(self, segment):
        self.applyScoreEntries(self.scoreEntriesMahalanobis(segment))

    def calculateScoreEuclidean(self, segment):
        self.applyScoreEntries(self.scoreEntriesEuclidean(segment))

    def calculateScoreEuclideanAndMahalanobis(self, segment):
        self.applyScoreEntries(self.scoreEntriesEuclideanAndMahalanobis(segment))

    def calculateScoreMGC(self, segment):
        self.applyScoreEntries(self.scoreEntriesMGC(segment))

    def checkCompatibility(self, booleanarray, max_height, max_width):
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
        self_pic_matrix[2:shape[0]+2, 2:shape[1]+2] = self.pic_connection_matrix
        pieces_to_check = self_pic_matrix.nonzero()
        score_dict = self.score_dict
        checkCompatibility = self.checkCompatibility
        compare_segment_piece_number = compare_segment.piece_number
        for x, y in zip(pieces_to_check[0], pieces_to_check[1]):
            if self_pic_matrix[x+1][y] == 0:
                score = 0
                number_of_sides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.DOWN, compare_segment_piece_number]
                if self_pic_matrix[x+2][y] != 0:  # check piece to right down and left
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+2][y].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x+1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x+1][y+1].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x+1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x+1][y-1].piece_number]
                    number_of_sides += 1
                score = score/number_of_sides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x+1, y] = compare_segment
                    temp_binary_matrix[x+1, y] = 1
                    if checkCompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setConnection(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)

            if self_pic_matrix[x-1][y] == 0:
                score = 0
                number_of_sides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.UP, compare_segment_piece_number]
                if self_pic_matrix[x-2][y] != 0:  # check piece to right down and left
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-2][y].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x-1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x-1][y+1].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x-1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x-1][y-1].piece_number]
                    number_of_sides += 1
                score = score/number_of_sides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x-1, y] = compare_segment
                    temp_binary_matrix[x-1, y] = 1
                    if checkCompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setConnection(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
            if self_pic_matrix[x][y+1] == 0:
                score = 0
                number_of_sides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.RIGHT, compare_segment_piece_number]
                if self_pic_matrix[x][y+2] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.RIGHT, self_pic_matrix[x][y+2].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x+1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+1][y+1].piece_number]
                    number_of_sides += 1

                if self_pic_matrix[x-1][y+1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-1][y+1].piece_number]
                    number_of_sides += 1
                score = score/number_of_sides

                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x, y+1] = compare_segment
                    temp_binary_matrix[x, y+1] = 1
                    if checkCompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setConnection(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
            if self_pic_matrix[x][y-1] == 0:
                score = 0
                number_of_sides = 1
                score += score_dict[self_pic_matrix[x][y].piece_number,
                                    JoinDirection.LEFT, compare_segment_piece_number]
                if self_pic_matrix[x][y-2] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.LEFT, self_pic_matrix[x][y-2].piece_number]
                    number_of_sides += 1
                if self_pic_matrix[x+1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.DOWN, self_pic_matrix[x+1][y-1].piece_number]
                    number_of_sides += 1
                if self_pic_matrix[x-1][y-1] != 0:
                    score += score_dict[compare_segment_piece_number,
                                        JoinDirection.UP, self_pic_matrix[x-1][y-1].piece_number]
                    number_of_sides += 1
                score = score/number_of_sides
                if score < best_connection_found_so_far.score:
                    temp_pic_matrix = copy(self_pic_matrix)
                    temp_binary_matrix = copy(self_binary_matrix)
                    temp_pic_matrix[x, y-1] = compare_segment
                    temp_binary_matrix[x, y-1] = 1
                    if checkCompatibility(temp_binary_matrix, self.max_height, self.max_width):
                        best_connection_found_so_far.setConnection(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
        return best_connection_found_so_far

    def calculateConnectionsKruskal(self, compare_segment, boost_priority_of_big_pieces_joining):
        if (self.component_id, compare_segment.component_id) in self.connections_dict:
            return self.connections_dict[(self.component_id, compare_segment.component_id)]
        score_dict = self.score_dict
        best_connection_found_so_far = self.best_connection_found_so_far
        own_data = self.kruskalComponentData()
        compare_data = compare_segment.kruskalComponentData()
        h1 = own_data["height"]
        w1 = own_data["width"]
        h2 = compare_data["height"]
        w2 = compare_data["width"]
        height_padded = h1+2*h2
        width_padded = w1+2*w2
        max_height = self.max_height
        max_width = self.max_width
        own_positions_padded = tuple(
            (row+h2, col+w2)
            for row, col in own_data["positions"]
        )
        own_position_set = set(own_positions_padded)
        own_piece_numbers = {
            (row+h2, col+w2): piece_number
            for (row, col), piece_number in own_data["piece_by_position"].items()
        }

        for x, y in self.kruskalCandidateOffsets(
                own_data,
                compare_data,
                h2,
                w2,
                height_padded,
                width_padded):
            compare_piece_numbers = {
                (row+x, col+y): piece_number
                for (row, col), piece_number
                in compare_data["piece_by_position"].items()
            }
            if any(position in own_position_set
                   for position in compare_piece_numbers):
                continue
            if not self.kruskalPlacementFits(
                    own_data,
                    compare_data,
                    x,
                    y,
                    h2,
                    w2,
                    max_height,
                    max_width):
                continue

            score = 0
            comparison_count = 0
            for row, col in own_positions_padded:
                node1 = own_piece_numbers[row, col]
                adjacent_piece = compare_piece_numbers.get((row, col+1))
                if adjacent_piece is not None:
                    comparison_count += 1
                    score += score_dict[node1, JoinDirection.RIGHT, adjacent_piece]
                adjacent_piece = compare_piece_numbers.get((row, col-1))
                if adjacent_piece is not None:
                    comparison_count += 1
                    score += score_dict[node1, JoinDirection.LEFT, adjacent_piece]
                adjacent_piece = compare_piece_numbers.get((row+1, col))
                if adjacent_piece is not None:
                    comparison_count += 1
                    score += score_dict[node1, JoinDirection.DOWN, adjacent_piece]
                adjacent_piece = compare_piece_numbers.get((row-1, col))
                if adjacent_piece is not None:
                    comparison_count += 1
                    score += score_dict[node1, JoinDirection.UP, adjacent_piece]
            if comparison_count == 0:
                continue
            if boost_priority_of_big_pieces_joining:
                score = score/((comparison_count*comparison_count)*0.5)
            else:
                score = score/comparison_count
            if score < best_connection_found_so_far.score:
                combined_pieces = zeros((height_padded, width_padded))
                combined_pointer = zeros(
                    (height_padded, width_padded), dtype="object")
                for row, col in own_data["positions"]:
                    combined_row = row+h2
                    combined_col = col+w2
                    combined_pieces[combined_row, combined_col] = 1
                    combined_pointer[combined_row, combined_col] = (
                        own_data["pic_matrix"][row, col]
                    )
                for row, col in compare_data["positions"]:
                    combined_row = row+x
                    combined_col = col+y
                    combined_pieces[combined_row, combined_col] = 1
                    combined_pointer[combined_row, combined_col] = (
                        compare_data["pic_matrix"][row, col]
                    )
                best_connection_found_so_far.setConnection(
                    combined_pointer,
                    compare_segment,
                    score,
                    self,
                    combined_pieces,
                )
        self.connections_dict[(
            self.component_id, compare_segment.component_id)] = best_connection_found_so_far
        return best_connection_found_so_far

    def kruskalComponentData(self):
        cached = self._kruskal_component_data
        if cached is not None and cached["component_id"] == self.component_id:
            return cached

        binary_matrix = self.binary_connection_matrix
        pic_matrix = self.pic_connection_matrix
        rows, cols = nonzero(binary_matrix)
        positions = tuple(
            (int(row), int(col))
            for row, col in zip(rows, cols)
        )
        position_set = set(positions)
        boundary_positions = set()
        for row, col in positions:
            for row_delta, col_delta in (
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0)):
                neighbor = (row+row_delta, col+col_delta)
                if neighbor not in position_set:
                    boundary_positions.add(neighbor)

        self._kruskal_component_data = {
            "component_id": self.component_id,
            "pic_matrix": pic_matrix,
            "height": binary_matrix.shape[0],
            "width": binary_matrix.shape[1],
            "positions": positions,
            "piece_by_position": {
                position: pic_matrix[position].piece_number
                for position in positions
            },
            "boundary_positions": tuple(sorted(boundary_positions)),
            "min_row": min(row for row, _col in positions),
            "max_row": max(row for row, _col in positions),
            "min_col": min(col for _row, col in positions),
            "max_col": max(col for _row, col in positions),
        }
        return self._kruskal_component_data

    def kruskalCandidateOffsets(
            self,
            own_data,
            compare_data,
            own_row_offset,
            own_col_offset,
            height_padded,
            width_padded):
        max_x = height_padded - compare_data["height"]
        max_y = width_padded - compare_data["width"]
        offsets = set()
        for neighbor_row, neighbor_col in own_data["boundary_positions"]:
            padded_neighbor_row = neighbor_row + own_row_offset
            padded_neighbor_col = neighbor_col + own_col_offset
            for compare_row, compare_col in compare_data["positions"]:
                x = padded_neighbor_row - compare_row
                y = padded_neighbor_col - compare_col
                if 0 <= x <= max_x and 0 <= y <= max_y:
                    offsets.add((x, y))
        return sorted(offsets)

    def kruskalPlacementFits(
            self,
            own_data,
            compare_data,
            x,
            y,
            own_row_offset,
            own_col_offset,
            max_height,
            max_width):
        min_row = min(own_row_offset + own_data["min_row"],
                      x + compare_data["min_row"])
        max_row = max(own_row_offset + own_data["max_row"],
                      x + compare_data["max_row"])
        min_col = min(own_col_offset + own_data["min_col"],
                      y + compare_data["min_col"])
        max_col = max(own_col_offset + own_data["max_col"],
                      y + compare_data["max_col"])
        return not (
            max_col-min_col > max_height
            or max_row-min_row > max_width
        )
