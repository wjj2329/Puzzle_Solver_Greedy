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


_DIRECTION_INDEX = {
    direction: index
    for index, direction in enumerate(JoinDirection)
}
_UP_INDEX = _DIRECTION_INDEX[JoinDirection.UP]
_DOWN_INDEX = _DIRECTION_INDEX[JoinDirection.DOWN]
_LEFT_INDEX = _DIRECTION_INDEX[JoinDirection.LEFT]
_RIGHT_INDEX = _DIRECTION_INDEX[JoinDirection.RIGHT]


def scoreValuesArray(score_dict):
    scalar_score_values = getattr(score_dict, "scalarScoreValues", None)
    if scalar_score_values is None:
        return None
    return scalar_score_values()


def scoreLookup(score_dict):
    score_value = getattr(score_dict, "scoreValue", None)
    if score_value is not None:
        return score_value

    def mappingScoreValue(own_number, direction, join_number):
        return score_dict[own_number, direction, join_number]

    return mappingScoreValue


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
        self.kruskal_connection_data = None

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

    def hasConnection(self):
        return (
            self.pic_connection_matrix is not None
            or self.kruskal_connection_data is not None
        )


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

    def calculateConnectionsKruskal(
            self,
            compare_segment,
            boost_priority_of_big_pieces_joining,
            defer_connection_matrices=False):
        connection_cache_key = (self.component_id, compare_segment.component_id)
        cached_connection = self.connections_dict.get(connection_cache_key)
        if cached_connection is not None:
            if not defer_connection_matrices:
                self.materializeKruskalConnection(cached_connection)
            return cached_connection
        score_dict = self.score_dict
        score_values = scoreValuesArray(score_dict)
        score_lookup = None if score_values is not None else scoreLookup(score_dict)
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
        own_shifted_data = self.kruskalShiftedComponentData(own_data, h2, w2)
        own_positions_padded = own_shifted_data["positions"]
        own_boundary_rows = own_shifted_data["boundary_piece_rows"]
        own_boundary_cols = own_shifted_data["boundary_piece_cols"]
        own_boundary_piece_numbers = own_shifted_data["boundary_piece_numbers"]
        own_position_set = own_shifted_data["position_set"]
        own_piece_numbers = own_shifted_data["piece_by_position"]
        compare_positions = compare_data["positions"]
        compare_piece_by_position = compare_data["piece_by_position"]
        compare_piece_number_grid = compare_data["piece_number_grid"]
        compare_is_single_piece = len(compare_positions) == 1
        own_is_single_piece = len(own_positions_padded) == 1
        if compare_is_single_piece:
            compare_piece_position = compare_positions[0]
            compare_piece_number = compare_piece_by_position[
                compare_piece_position]
        if own_is_single_piece:
            own_piece_position = own_positions_padded[0]
            own_piece_number = own_piece_numbers[own_piece_position]
        best_connection_offset = None

        for x, y in self.kruskalCandidateOffsets(
                own_data,
                compare_data,
                h2,
                w2,
                height_padded,
                width_padded):
            if compare_is_single_piece:
                shifted_compare_position = (
                    compare_piece_position[0] + x,
                    compare_piece_position[1] + y,
                )
                if shifted_compare_position in own_position_set:
                    continue
            elif own_is_single_piece:
                if (
                        own_piece_position[0] - x,
                        own_piece_position[1] - y,
                ) in compare_piece_by_position:
                    continue
            else:
                if self.kruskalComponentsOverlap(
                        own_data,
                        compare_data,
                        x,
                        y,
                        h2,
                        w2):
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

            if compare_is_single_piece:
                score, comparison_count = self.kruskalSingleCompareScore(
                    own_piece_numbers,
                    shifted_compare_position,
                    compare_piece_number,
                    score_lookup,
                    score_values,
                )
            elif own_is_single_piece:
                score, comparison_count = self.kruskalSingleOwnScore(
                    own_piece_number,
                    own_piece_position,
                    compare_piece_by_position,
                    x,
                    y,
                    score_lookup,
                    score_values,
                )
            else:
                score, comparison_count = self.kruskalScoreGridPieces(
                    own_boundary_rows,
                    own_boundary_cols,
                    own_boundary_piece_numbers,
                    compare_piece_number_grid,
                    x,
                    y,
                    score_lookup,
                    score_values,
                )
            if comparison_count == 0:
                continue
            if boost_priority_of_big_pieces_joining:
                score = score/((comparison_count*comparison_count)*0.5)
            else:
                score = score/comparison_count
            if score < best_connection_found_so_far.score:
                best_connection_found_so_far.second_best_score = (
                    best_connection_found_so_far.score
                )
                best_connection_found_so_far.score = score
                best_connection_found_so_far.own_segment = self
                best_connection_found_so_far.join_segment = compare_segment
                best_connection_offset = (x, y)
        if best_connection_offset is not None:
            x, y = best_connection_offset
            best_connection_found_so_far.kruskal_connection_data = (
                own_data,
                compare_data,
                x,
                y,
                h2,
                w2,
                height_padded,
                width_padded,
            )
            if not defer_connection_matrices:
                self.materializeKruskalConnection(best_connection_found_so_far)
        self.connections_dict[connection_cache_key] = best_connection_found_so_far
        return best_connection_found_so_far

    def materializeKruskalConnection(self, connection):
        if connection.pic_connection_matrix is not None:
            return
        if connection.kruskal_connection_data is None:
            return
        combined_pointer, combined_pieces = self.kruskalConnectionMatrices(
            *connection.kruskal_connection_data,
        )
        connection.pic_connection_matrix = combined_pointer
        connection.binary_connection_matrix = combined_pieces
        connection.kruskal_connection_data = None

    def kruskalShiftedComponentData(self, component_data, row_offset, col_offset):
        shifted_cache = component_data.setdefault("shifted", {})
        cache_key = (row_offset, col_offset)
        shifted_data = shifted_cache.get(cache_key)
        if shifted_data is not None:
            return shifted_data

        positions = tuple(
            (row+row_offset, col+col_offset)
            for row, col in component_data["positions"]
        )
        shifted_data = {
            "positions": positions,
            "boundary_piece_positions": tuple(
                (row+row_offset, col+col_offset)
                for row, col in component_data["boundary_piece_positions"]
            ),
            "boundary_piece_rows": tuple(
                row+row_offset
                for row in component_data["boundary_piece_rows"]
            ),
            "boundary_piece_cols": tuple(
                col+col_offset
                for col in component_data["boundary_piece_cols"]
            ),
            "boundary_piece_numbers": component_data[
                "boundary_piece_numbers"],
            "position_set": set(positions),
            "piece_by_position": {
                (row+row_offset, col+col_offset): piece_number
                for (row, col), piece_number
                in component_data["piece_by_position"].items()
            },
        }
        shifted_cache[cache_key] = shifted_data
        return shifted_data

    def kruskalSingleCompareScore(
            self,
            own_piece_numbers,
            compare_position,
            compare_piece_number,
            score_lookup,
            score_values):
        row, col = compare_position
        score = 0
        comparison_count = 0

        adjacent_piece = own_piece_numbers.get((row-1, col))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(adjacent_piece, JoinDirection.DOWN,
                                      compare_piece_number)
            else:
                score += score_values[adjacent_piece, _DOWN_INDEX,
                                      compare_piece_number]
        adjacent_piece = own_piece_numbers.get((row, col-1))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(adjacent_piece, JoinDirection.RIGHT,
                                      compare_piece_number)
            else:
                score += score_values[adjacent_piece, _RIGHT_INDEX,
                                      compare_piece_number]
        adjacent_piece = own_piece_numbers.get((row, col+1))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(adjacent_piece, JoinDirection.LEFT,
                                      compare_piece_number)
            else:
                score += score_values[adjacent_piece, _LEFT_INDEX,
                                      compare_piece_number]
        adjacent_piece = own_piece_numbers.get((row+1, col))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(adjacent_piece, JoinDirection.UP,
                                      compare_piece_number)
            else:
                score += score_values[adjacent_piece, _UP_INDEX,
                                      compare_piece_number]

        return score, comparison_count

    def kruskalSingleOwnScore(
            self,
            own_piece_number,
            own_position,
            compare_piece_by_position,
            compare_row_offset,
            compare_col_offset,
            score_lookup,
            score_values):
        row, col = own_position
        score = 0
        comparison_count = 0

        adjacent_piece = compare_piece_by_position.get(
            (row - compare_row_offset, col + 1 - compare_col_offset))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(own_piece_number, JoinDirection.RIGHT,
                                      adjacent_piece)
            else:
                score += score_values[own_piece_number, _RIGHT_INDEX,
                                      adjacent_piece]
        adjacent_piece = compare_piece_by_position.get(
            (row - compare_row_offset, col - 1 - compare_col_offset))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(own_piece_number, JoinDirection.LEFT,
                                      adjacent_piece)
            else:
                score += score_values[own_piece_number, _LEFT_INDEX,
                                      adjacent_piece]
        adjacent_piece = compare_piece_by_position.get(
            (row + 1 - compare_row_offset, col - compare_col_offset))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(own_piece_number, JoinDirection.DOWN,
                                      adjacent_piece)
            else:
                score += score_values[own_piece_number, _DOWN_INDEX,
                                      adjacent_piece]
        adjacent_piece = compare_piece_by_position.get(
            (row - 1 - compare_row_offset, col - compare_col_offset))
        if adjacent_piece is not None:
            comparison_count += 1
            if score_values is None:
                score += score_lookup(own_piece_number, JoinDirection.UP,
                                      adjacent_piece)
            else:
                score += score_values[own_piece_number, _UP_INDEX,
                                      adjacent_piece]

        return score, comparison_count

    def kruskalScoreGridPieces(
            self,
            own_rows,
            own_cols,
            own_piece_numbers,
            compare_piece_number_grid,
            compare_row_offset,
            compare_col_offset,
            score_lookup,
            score_values):
        score = 0
        comparison_count = 0
        compare_height, compare_width = compare_piece_number_grid.shape
        for index, row in enumerate(own_rows):
            col = own_cols[index]
            node1 = own_piece_numbers[index]
            compare_row = row - compare_row_offset

            compare_col = col + 1 - compare_col_offset
            if (
                    0 <= compare_row < compare_height
                    and 0 <= compare_col < compare_width):
                adjacent_piece = compare_piece_number_grid[
                    compare_row, compare_col]
            else:
                adjacent_piece = 0
            if adjacent_piece != 0:
                comparison_count += 1
                if score_values is None:
                    adjacent_piece = int(adjacent_piece)
                    score += score_lookup(node1, JoinDirection.RIGHT,
                                          adjacent_piece)
                else:
                    score += score_values[node1, _RIGHT_INDEX, adjacent_piece]

            compare_col = col - 1 - compare_col_offset
            if (
                    0 <= compare_row < compare_height
                    and 0 <= compare_col < compare_width):
                adjacent_piece = compare_piece_number_grid[
                    compare_row, compare_col]
            else:
                adjacent_piece = 0
            if adjacent_piece != 0:
                comparison_count += 1
                if score_values is None:
                    adjacent_piece = int(adjacent_piece)
                    score += score_lookup(node1, JoinDirection.LEFT,
                                          adjacent_piece)
                else:
                    score += score_values[node1, _LEFT_INDEX, adjacent_piece]

            compare_row = row + 1 - compare_row_offset
            compare_col = col - compare_col_offset
            if (
                    0 <= compare_row < compare_height
                    and 0 <= compare_col < compare_width):
                adjacent_piece = compare_piece_number_grid[
                    compare_row, compare_col]
            else:
                adjacent_piece = 0
            if adjacent_piece != 0:
                comparison_count += 1
                if score_values is None:
                    adjacent_piece = int(adjacent_piece)
                    score += score_lookup(node1, JoinDirection.DOWN,
                                          adjacent_piece)
                else:
                    score += score_values[node1, _DOWN_INDEX, adjacent_piece]

            compare_row = row - 1 - compare_row_offset
            compare_col = col - compare_col_offset
            if (
                    0 <= compare_row < compare_height
                    and 0 <= compare_col < compare_width):
                adjacent_piece = compare_piece_number_grid[
                    compare_row, compare_col]
            else:
                adjacent_piece = 0
            if adjacent_piece != 0:
                comparison_count += 1
                if score_values is None:
                    adjacent_piece = int(adjacent_piece)
                    score += score_lookup(node1, JoinDirection.UP,
                                          adjacent_piece)
                else:
                    score += score_values[node1, _UP_INDEX, adjacent_piece]
        return score, comparison_count

    def kruskalComponentsOverlap(
            self,
            own_data,
            compare_data,
            compare_row_offset,
            compare_col_offset,
            own_row_offset,
            own_col_offset):
        own_binary_matrix = own_data["binary_matrix"]
        own_height = own_data["height"]
        own_width = own_data["width"]
        for row, col in compare_data["positions"]:
            own_row = row + compare_row_offset - own_row_offset
            own_col = col + compare_col_offset - own_col_offset
            if (
                    0 <= own_row < own_height
                    and 0 <= own_col < own_width
                    and own_binary_matrix[own_row, own_col] != 0):
                return True
        return False

    def kruskalConnectionMatrices(
            self,
            own_data,
            compare_data,
            compare_row_offset,
            compare_col_offset,
            own_row_offset,
            own_col_offset,
            height_padded,
            width_padded):
        combined_pieces = zeros((height_padded, width_padded))
        combined_pointer = zeros((height_padded, width_padded), dtype="object")
        for row, col in own_data["positions"]:
            combined_row = row+own_row_offset
            combined_col = col+own_col_offset
            combined_pieces[combined_row, combined_col] = 1
            combined_pointer[combined_row, combined_col] = (
                own_data["pic_matrix"][row, col]
            )
        for row, col in compare_data["positions"]:
            combined_row = row+compare_row_offset
            combined_col = col+compare_col_offset
            combined_pieces[combined_row, combined_col] = 1
            combined_pointer[combined_row, combined_col] = (
                compare_data["pic_matrix"][row, col]
            )
        return combined_pointer, combined_pieces

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
        piece_number_grid = np.zeros(binary_matrix.shape, dtype=np.int32)
        piece_by_position = {}
        for position in positions:
            piece_number = pic_matrix[position].piece_number
            piece_by_position[position] = piece_number
            piece_number_grid[position] = piece_number
        position_set = set(positions)
        boundary_positions = set()
        boundary_piece_positions = []
        for row, col in positions:
            is_boundary_piece = False
            for row_delta, col_delta in (
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0)):
                neighbor = (row+row_delta, col+col_delta)
                if neighbor not in position_set:
                    boundary_positions.add(neighbor)
                    is_boundary_piece = True
            if is_boundary_piece:
                boundary_piece_positions.append((row, col))

        self._kruskal_component_data = {
            "component_id": self.component_id,
            "pic_matrix": pic_matrix,
            "height": binary_matrix.shape[0],
            "width": binary_matrix.shape[1],
            "binary_matrix": binary_matrix,
            "piece_number_grid": piece_number_grid,
            "positions": positions,
            "piece_by_position": piece_by_position,
            "boundary_positions": tuple(sorted(boundary_positions)),
            "boundary_piece_positions": tuple(boundary_piece_positions),
            "boundary_piece_rows": tuple(
                row for row, _col in boundary_piece_positions),
            "boundary_piece_cols": tuple(
                col for _row, col in boundary_piece_positions),
            "boundary_piece_numbers": tuple(
                piece_by_position[position]
                for position in boundary_piece_positions
            ),
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

        if len(compare_data["positions"]) == 1:
            compare_row, compare_col = compare_data["positions"][0]
            return [
                (x, y)
                for x, y in (
                    (
                        neighbor_row + own_row_offset - compare_row,
                        neighbor_col + own_col_offset - compare_col,
                    )
                    for neighbor_row, neighbor_col
                    in own_data["boundary_positions"]
                )
                if 0 <= x <= max_x and 0 <= y <= max_y
            ]

        if len(own_data["positions"]) == 1:
            own_row, own_col = own_data["positions"][0]
            padded_own_row = own_row + own_row_offset
            padded_own_col = own_col + own_col_offset
            return sorted(
                (x, y)
                for x, y in (
                    (
                        padded_own_row - boundary_row,
                        padded_own_col - boundary_col,
                    )
                    for boundary_row, boundary_col
                    in compare_data["boundary_positions"]
                )
                if 0 <= x <= max_x and 0 <= y <= max_y
            )

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
