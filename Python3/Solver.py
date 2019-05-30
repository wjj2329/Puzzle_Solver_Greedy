import argparse
import scipy as sp
import numpy as np
import random
import sys
import time
import tkinter
import math
from copy import copy
from imageio import imread, imsave
from enum import Enum
from scipy.ndimage.morphology import binary_dilation
from PIL import ImageTk, Image
from skimage import io, color
from numpy import logical_and, zeros, nonzero, argwhere, delete, asarray
from numpy import sum as numpySum
from numpy import all as numpyAll
from numpy.linalg import norm


class JoinDirection(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class CompareWithOtherSegments(Enum):
    ONLY_BEST = 1
    COMPARE_WITH_SECOND = 2


class ScoreAlgorithum(Enum):
    EUCLIDEAN = 1
    MAHALANOBIS = 2
    GRADIENT_EUCLIDEAN = 3


class ColorType(Enum):
    RGB = 1
    LAB = 2


class AssemblyType(Enum):
    KRUSKAL = 1
    PRIM = 2


class BestConnection:
    pic_connection_matix = None
    join_segment = None
    score = sys.maxsize
    own_segment = None
    binary_connection_matrix = None
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
        # not this but something to use this ratio with the score
        elif compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
            return (self.score*(1/(self.score/self.second_best_score))) < (otherConnection.score*(1/(otherConnection.score/otherConnection.second_best_score)))

    def setNodeContents(self, my_list):
        my_list.remove(self.own_segment)
        self.own_segment.pic_connection_matix = self.pic_connection_matix
        self.own_segment.binary_connection_matrix = self.binary_connection_matrix
        my_list.append(self.own_segment)

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


# malhnobis and different color spaces to try out.

class Segment:
    myownNumber = 0
    pic_matrix = None
    score_dict = {}
    connections_dict = {}
    connections_first_time = []
    binary_connection_matrix = asarray([[1, 0], [0, 0]])
    pic_connection_matix = None
    max_width = 0
    max_height = 0
    best_connection_found_so_far = BestConnection()
    piece_number = -1
    mal_data = None

    def __init__(self, pic_matrix, max_width, max_height, piece_number, myownNumber, score_dict, mal_data=None):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = asarray([[self, 0], [0, 0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number
        self.myownNumber = myownNumber
        self.score_dict = score_dict
        self.mal_data = mal_data

    def euclideanDistance(self, a, b):
        a = a[0].astype(np.float64)  # underflows would occur without this
        b = b[0].astype(np.float64)
        temp = sum(np.linalg.norm(x - y) for x, y in zip(a, b))
        return temp

    def mahalanobisDistance(self, a, a2, z, z2):
        a = a[0].astype(np.int16)  # underflows would occur without this
        a2 = a2[0].astype(np.int16)
        z = z[0].astype(np.int16)
        z2 = z2[0].astype(np.int16)

        cov = np.linalg.pinv(sp.cov((a).T))
        cov2 = np.linalg.pinv(sp.cov((z).T))

        red_average_1 = np.average(a[:, 0]-a2[:, 0])
        green_average_1 = np.average(a[:, 1]-a2[:, 1])
        blue_average_1 = np.average(a[:, 2]-a2[:, 2])

        red_average_2 = np.average(z[:, 0]-z2[:, 0])
        green_average_2 = np.average(z[:, 1]-z2[:, 1])
        blue_average_2 = np.average(z[:, 2]-z2[:, 2])

        score = 0.0
        testr1 = a[:, 0]-z[:, 0]
        testg1 = a[:, 1]-z[:, 1]
        testb1 = a[:, 2]-z[:, 2]

        testr2 = z[:, 0]-a[:, 0]
        testg2 = z[:, 1]-a[:, 1]
        testb2 = z[:, 2]-a[:, 2]

        for i in range(len(a)):
            mymatrix = np.matrix(
                [testr1[i]-red_average_1, testg1[i]-green_average_1, testb1[i]-blue_average_1])
            mymatrix2 = np.matrix(
                [testr2[i]-red_average_2, testg2[i]-green_average_2, testb2[i]-blue_average_2])
            score += math.sqrt(abs(mymatrix*cov2*mymatrix.T))
            score += math.sqrt(abs(mymatrix2*cov*mymatrix2.T))
        return score

    def calculateScoreGradientEuclideanDistance(self, segment):
        score_dict = self.score_dict
        euclideanDistance = self.euclideanDistance
        size = segment.pic_matrix.shape[0]
        pic_matrix = self.pic_matrix
        self_top1 = pic_matrix[0:1, :, :]
        self_top2 = pic_matrix[1:2, :, :]
        self_left1 = np.rot90(pic_matrix[:, 0:1, :])
        self_left2 = np.rot90(pic_matrix[:, 1:2, :])
        self_bottom1 = pic_matrix[size - 1:size, :, :]
        self_bottom2 = pic_matrix[size-2:size-1, :, :]
        self_right1 = np.rot90(pic_matrix[:, size - 1:size, :])
        self_right2 = np.rot90(pic_matrix[:, size - 2:size-1, :])

        segment_matrix = segment.pic_matrix
        compare_top1 = segment_matrix[0:1, :, :]
        compare_top2 = segment_matrix[1:2, :, :]
        compare_left1 = np.rot90(segment_matrix[:, 0:1, :])
        compare_left2 = np.rot90(segment_matrix[:, 1:2, :])
        compare_bottom1 = segment_matrix[size - 1:size, :, :]
        compare_bottom2 = segment_matrix[size - 2:size-1, :, :]
        compare_right1 = np.rot90(segment_matrix[:, size - 1:size, :])
        compare_right2 = np.rot90(segment_matrix[:, size-2:size-1, :])

        own_number = self.piece_number
        join_number = segment.piece_number

        self_top = self_top1-self_top2
        self_left = self_left1-self_left2
        self_bottom = self_bottom1-self_bottom2
        self_right = self_right1-self_right2

        compare_bottom = compare_bottom1-compare_bottom2
        compare_top = compare_top1-compare_top2
        compare_left = compare_left1-compare_left2
        compare_right = compare_right1-compare_right2

        score_dict[own_number, JoinDirection.UP,
                   join_number] = euclideanDistance(self_top, compare_bottom)
        score_dict[own_number, JoinDirection.DOWN,
                   join_number] = euclideanDistance(self_bottom, compare_top)
        score_dict[own_number, JoinDirection.LEFT,
                   join_number] = euclideanDistance(self_left, compare_right)
        score_dict[own_number, JoinDirection.RIGHT,
                   join_number] = euclideanDistance(self_right, compare_left)

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


# http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
# https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
# https://www.python.org/dev/peps/pep-0371/ use this to make it faster

    def calculateScoreMahalonbis(self, segment):
        size = segment.pic_matrix.shape[0]
        pic_matrix = self.pic_matrix
        self_top = pic_matrix[0:1, :, :]
        self_top2 = pic_matrix[1:2, :, :]
        self_left = np.rot90(pic_matrix[:, 0:1, :])
        self_left2 = np.rot90(pic_matrix[:, 1:2, :])
        self_bottom = pic_matrix[size - 1:size, :, :]
        self_bottom2 = pic_matrix[size-2:size-1, :, :]
        self_right = np.rot90(pic_matrix[:, size - 1:size, :])
        self_right2 = np.rot90(pic_matrix[:, size - 2:size-1, :])

        segment_matrix = segment.pic_matrix
        compare_top = segment_matrix[0:1, :, :]
        compare_top2 = segment_matrix[1:2, :, :]
        compare_left = np.rot90(segment_matrix[:, 0:1, :])
        compare_left2 = np.rot90(segment_matrix[:, 1:2, :])
        compare_bottom = segment_matrix[size - 1:size, :, :]
        compare_bottom2 = segment_matrix[size - 2:size-1, :, :]
        compare_right = np.rot90(segment_matrix[:, size - 1:size, :])
        compare_right2 = np.rot90(segment_matrix[:, size-2:size-1, :])

        own_number = self.piece_number
        join_number = segment.piece_number
        self.score_dict[own_number, JoinDirection.UP,
                        join_number] = self.mahalanobisDistance(self_top, self_top2, compare_bottom, compare_bottom2)
        self.score_dict[own_number, JoinDirection.DOWN,
                        join_number] = self.mahalanobisDistance(self_bottom, self_bottom2, compare_top, compare_top2)
        self.score_dict[own_number, JoinDirection.LEFT,
                        join_number] = self.mahalanobisDistance(self_left, self_left2, compare_right, compare_right2)
        self.score_dict[own_number, JoinDirection.RIGHT,
                        join_number] = self.mahalanobisDistance(self_right, self_right2, compare_left, compare_left2)

        self.score_dict[join_number, JoinDirection.DOWN,
                        own_number] = self.score_dict[own_number, JoinDirection.UP,
                                                      join_number]
        self.score_dict[join_number, JoinDirection.UP,
                        own_number] = self.score_dict[own_number, JoinDirection.DOWN,
                                                      join_number]
        self.score_dict[join_number, JoinDirection.RIGHT,
                        own_number] = self.score_dict[own_number, JoinDirection.LEFT,
                                                      join_number]
        self.score_dict[join_number, JoinDirection.LEFT,
                        own_number] = self.score_dict[own_number, JoinDirection.RIGHT,
                                                      join_number]

    def calculateScoreEuclidean(self, segment):
        size = segment.pic_matrix.shape[0]
        score_dict = self.score_dict
        euclideanDistance = self.euclideanDistance

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
                   join_number] = euclideanDistance(self_top, compare_bottom)
        score_dict[own_number, JoinDirection.DOWN,
                   join_number] = euclideanDistance(self_bottom, compare_top)
        score_dict[own_number, JoinDirection.LEFT,
                   join_number] = euclideanDistance(self_left, compare_right)
        score_dict[own_number, JoinDirection.RIGHT,
                   join_number] = euclideanDistance(self_right, compare_left)

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

    # make sure this works as intended

    def checkforcompatibility(self, booleanarray):
        whattokeep = nonzero(booleanarray)
        smallestx1 = min(nonzero(booleanarray)[1])
        smallesty1 = min(nonzero(booleanarray)[0])
        biggestx1 = max(nonzero(booleanarray)[1])
        biggesty1 = max(nonzero(booleanarray)[0])
        biggest = biggestx1-smallestx1+1
        if(biggesty1 - smallesty1 + 1 > biggest):
            biggest = biggesty1 - smallesty1 + 1
        storeing = zeros((biggest, biggest), dtype="object")
        for y in range(0, len(whattokeep[0])):
            pair = [whattokeep[0][y], whattokeep[1][y]]
        storeing[pair[0] - smallesty1][pair[1] -
                                       smallestx1] = booleanarray[pair[0]][pair[1]]
        temp = storeing
        if temp.shape[0] > self.max_height or temp.shape[1] > self.max_width:
            return False
        return True

    # verify this is correct
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
                    if checkforcompatibility(temp_binary_matrix):
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
                    if checkforcompatibility(temp_binary_matrix):
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
                    if checkforcompatibility(temp_binary_matrix):
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
                    if checkforcompatibility(temp_binary_matrix):
                        best_connection_found_so_far.setThings(
                            temp_pic_matrix, compare_segment, score, self, temp_binary_matrix)
        return best_connection_found_so_far

    # rename these bad varaible names
    def calculateConnectionsKruskal(self, compare_segment):
        if (self.myownNumber, compare_segment.myownNumber) in self.connections_dict:
            return self.connections_dict[(self.myownNumber, compare_segment.myownNumber)]
        score_dict = self.score_dict
        best_connection_found_so_far = self.best_connection_found_so_far
        h1 = self.binary_connection_matrix.shape[0]
        w1 = self.binary_connection_matrix.shape[1]
        h2 = compare_segment.binary_connection_matrix.shape[0]
        w2 = compare_segment.binary_connection_matrix.shape[1]
        pad_with_piece1 = zeros((h1+2*h2, w1+2*w2))
        pad_with_piece1[h2:(h2+h1), w2:(w2+w1)] = self.binary_connection_matrix
        dilation_mask = asarray([[0, 1, 0], [1, 1, 1, ], [0, 1, 0]])
        result = binary_dilation(
            input=pad_with_piece1, structure=dilation_mask)
        neighboring_connections = result - pad_with_piece1
        for x in range(h1+2*h2-(h2-1)):
            for y in range(w1+2*w2-(w2-1)):
                pad_with_piece2 = zeros(neighboring_connections.shape)
                pad_with_piece2[x:(x+h2), y:(y+w2)
                                ] = compare_segment.binary_connection_matrix
                connect_map = logical_and(
                    neighboring_connections, pad_with_piece2)
                overlap_map = logical_and(pad_with_piece1, pad_with_piece2)
                has_connections = numpySum(connect_map[:]) > 0
                has_overlap = numpySum(overlap_map[:]) > 0
                combined_pieces = pad_with_piece1+pad_with_piece2
                if has_connections and not has_overlap and self.checkforcompatibility(combined_pieces):
                    store = nonzero(pad_with_piece1)
                    score = 0
                    numofcompar = 0
                    padded1_pointer = zeros(
                        (h1+2*h2, w1+2*w2), dtype="object")
                    padded1_pointer[h2:(h2+h1), w2:(w2+w1)
                                    ] = self.pic_connection_matix
                    temp_pointer = zeros((h1+2*h2, w1+2*w2), dtype="object")
                    temp_pointer[x:(h2+x), y:(w2+y)
                                 ] = compare_segment.pic_connection_matix
                    stuff = temp_pointer.nonzero()
                    for q, w in zip(stuff[0], stuff[1]):
                        padded1_pointer[q][w] = temp_pointer[q, w]
                    for d, h in zip(store[0], store[1]):
                        if pad_with_piece2[d][h+1] == 1:
                            node1 = padded1_pointer[d, h]
                            node2 = padded1_pointer[d, h+1]
                            numofcompar += 1
                            score += score_dict[node1.piece_number,
                                                JoinDirection.RIGHT, node2.piece_number]
                        if pad_with_piece2[d][h-1] == 1:
                            node1 = padded1_pointer[d, h]
                            node2 = padded1_pointer[d, h-1]
                            numofcompar += 1
                            score += score_dict[node1.piece_number,
                                                JoinDirection.LEFT, node2.piece_number]
                        if pad_with_piece2[d+1][h] == 1:
                            node1 = padded1_pointer[d, h]
                            node2 = padded1_pointer[d+1, h]
                            numofcompar += 1
                            score += score_dict[node1.piece_number,
                                                JoinDirection.DOWN, node2.piece_number]
                        if pad_with_piece2[d-1][h] == 1:
                            node1 = padded1_pointer[d, h]
                            node2 = padded1_pointer[d-1, h]
                            numofcompar += 1
                            score += score_dict[node1.piece_number,
                                                JoinDirection.UP, node2.piece_number]
                    score = score/numofcompar
                    if score < best_connection_found_so_far.score:
                        best_connection_found_so_far.setThings(
                            padded1_pointer, compare_segment, score, self, combined_pieces)
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


def breakUpImage(image, length, save_segments, colortype):
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
    mal_dict = {}
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[picX: picX+length, picY: picY+length, :]
            append(Segment(save, num_of_pieces_width,
                           num_of_pieces_height, piece_num, piece_num, score_dict, mal_dict))
            piece_num += 1
            if save_segments:
                if colortype == ColorType.RGB:
                    imsave(str(x)+"_"+str(y)+".png", save)
                if colortype == ColorType.LAB:
                    imsave(str(x)+"_"+str(y)+".png", color.lab2rgb(save))
            picY += length
        picX += length
        picY = 0
    return segments


def calculateScores(segment_list, score_algorithum):
    for index, segment1 in enumerate(segment_list):
        print("caculating score for segment ", segment1.piece_number)
        for segment2 in segment_list[index+1:]:
            if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
                segment1.calculateScoreEuclidean(segment2)
            if score_algorithum == ScoreAlgorithum.MAHALANOBIS:
                segment1.calculateScoreMahalonbis(segment2)
            if score_algorithum ==ScoreAlgorithum.GRADIENT_EUCLIDEAN:
                segment1.calculateScoreGradientEuclideanDistance(segment2)    


def findBestConnectionKruskal(segment_list, compare_type):
    best_so_far = BestConnection()
    for index, segment1 in enumerate(segment_list):
        for segment2 in segment_list[index+1:]:
            segment1.best_connection_found_so_far = BestConnection()
            temp = segment1.calculateConnectionsKruskal(segment2)
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


# TODO no idea how to find the best one to start,  will do random for now! Myabe find piece with best connections
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
        connection.connections_dict = {}


def saveImage(best_connection, piece_size, round, colortype):
    pic_locations = best_connection.binary_connection_matrix.nonzero()
    sizex = (max(pic_locations[0])-min(pic_locations[0]))+1
    sizey = (max(pic_locations[1])-min(pic_locations[1]))+1
    biggest_dim = sizex if sizex > sizey else sizey
    new_image = zeros((biggest_dim*piece_size, biggest_dim*piece_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matix[pic_locations[0]
                                                                 [x], pic_locations[1][x]].pic_matrix
        x1 = (pic_locations[0][x]-min(pic_locations[0]))*piece_size
        y1 = (pic_locations[1][x]-min(pic_locations[1]))*piece_size
        new_image[x1:x1+piece_size, y1:y1+piece_size, :] = piece_to_assemble
    if colortype == ColorType.LAB:
        new_image = color.lab2rgb(new_image)
    imageName = "round"+str(round)+".png"
    imsave(imageName, new_image)
    return imageName

# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?


def main():
    start_time = time.time()  # set up variables
    # parser = setUpArguments()
    picture_file_name = "william.png"  # parser.inputpic
    length = 240  # parser.length
    save_segments = True  # parser.savepieces
    image = imread(picture_file_name)  # parser.inputpic
    save_assembly_to_disk = True  # parser.saveassembly:
    show_building_animation = True  # parser.showanimation
    colorType = ColorType.RGB
    assemblyType = AssemblyType.KRUSKAL
    scoreType = ScoreAlgorithum.GRADIENT_EUCLIDEAN
    compareType = CompareWithOtherSegments.ONLY_BEST
    show_print_statements = True

    if colorType == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(image, length, save_segments, colorType)
    calculateScores(segment_list, scoreType)
    elapsed_time_secs = time.time() - start_time
    if show_print_statements:
        print("Calculate scores took: %s secs " % elapsed_time_secs)
    window, w = None, None
    if show_building_animation:
        window = tkinter.Tk()
        window.title("Picture")
        img = ImageTk.PhotoImage(Image.open(picture_file_name))
        w = tkinter.Label(window, image=img)
    random.shuffle(segment_list)
    round = 0
    original_size = len(segment_list)
    root = None
    if assemblyType == AssemblyType.PRIM:
        root = findBestRootSegment(segment_list)
    while len(segment_list) > 1:
        best_connection = None
        if assemblyType == AssemblyType.KRUSKAL:
            best_connection = findBestConnectionKruskal(
                segment_list, compareType)
        if assemblyType == AssemblyType.PRIM:
            best_connection = findBestConnectionPrim(
                segment_list, root, compareType)
        best_connection.stripZeros()
        best_connection.own_segment.binary_connection_matrix = best_connection.binary_connection_matrix
        best_connection.own_segment.pic_connection_matix = best_connection.pic_connection_matix
        best_connection.own_segment.myownNumber += original_size
        segment_list.remove(best_connection.join_segment)
        root = best_connection.own_segment
        if save_assembly_to_disk:
            updated_picture = ImageTk.PhotoImage(
                Image.open(saveImage(best_connection, length, round, colorType)))
            w.configure(image=updated_picture)
            w.image = updated_picture
            w.pack(side="bottom", fill="both", expand="no")
            window.update()
        round += 1
        if show_print_statements == True:
            print("for round ", round, " i get score of ", best_connection.score, "the ratio for first to second best is ",
                  best_connection.score/best_connection.second_best_score, " it took ", time.time()-start_time)
    if show_print_statements == True:
        elapsed_time_secs = time.time() - start_time
        print("Execution took: %s secs " % elapsed_time_secs)


if __name__ == '__main__':
    main()
