import argparse
import numpy as np
import random
from copy import copy
import sys
import time
from datetime import timedelta
from imageio import imread, imsave
from enum import Enum
from scipy.ndimage.morphology import binary_dilation
import tkinter
from PIL import ImageTk, Image
from skimage import io, color
import scipy as sp
from numpy import logical_and, zeros, nonzero, argwhere, delete, asarray
from numpy import sum as numpySum
from numpy import all as numpyAll


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

    def __init__(self, pic_matrix, max_width, max_height, piece_number, myownNumber, score_dict):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = asarray([[self, 0], [0, 0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number
        self.myownNumber = myownNumber
        self.score_dict=score_dict

    def euclideanDistance(self, a, b):
        a = a[0].astype(np.float64)  # underflows would occur without this
        b = b[0].astype(np.float64)
        temp = sum([np.linalg.norm(x - y) for x, y in zip(a, b)])
        return temp

    def mahalanobisDistance(self, a, a2, z, z2, piece2Num, direction):
        a = a[0].astype(np.int16)  # underflows would occur without this
        a2 = a2[0].astype(np.int16)
        z = z[0].astype(np.int16)
        z2 = z2[0].astype(np.int16)
        covariance_piece1 = zeros([3, 3])
        covariance_piece2 = zeros([3, 3])

        redav1_piece1 = np.average(a[:, 0])  # extract red blue and green
        greenav1_piece1 = np.average(a[:, 1])
        blueav1_piece1 = np.average(a[:, 2])
        redav2_piece1 = np.average(a2[:, 0])
        greenav2_piece1 = np.average(a2[:, 1])
        blueav2_piece1 = np.average(a2[:, 2])

        redav1_piece2 = np.average(z[:, 0])  # extract red blue and green
        greenav1_piece2 = np.average(z[:, 1])
        blueav1_piece2 = np.average(z[:, 2])
        redav2_piece2 = np.average(z2[:, 0])
        greenav2_piece2 = np.average(z2[:, 1])
        blueav2_piece2 = np.average(z2[:, 2])

        r1_piece1 = a[:, 0]-redav1_piece1
        r2_piece1 = a2[:, 0]-redav2_piece1
        g1_piece1 = a[:, 1]-greenav1_piece1
        g2_piece1 = a2[:, 1]-greenav2_piece1
        b1_piece1 = a[:, 2]-blueav1_piece1
        b2_piece1 = a2[:, 2]-blueav2_piece1
        '''
        size_piece1 = a2[:, 2].size
        size_piece1 = size_piece1/1.0
        r1r2_piece1 = np.dot(r1_piece1, r2_piece1)/size_piece1
        r1g2_piece1 = np.dot(r1_piece1, g2_piece1)/size_piece1
        r1b2_piece1 = np.dot(r1_piece1, b2_piece1)/size_piece1

        g1r2_piece1 = np.dot(g1_piece1, r2_piece1)/size_piece1
        g1g2_piece1 = np.dot(g1_piece1, g2_piece1)/size_piece1
        g1b2_piece1 = np.dot(g1_piece1, b2_piece1)/size_piece1

        b1r2_piece1 = np.dot(b1_piece1, r2_piece1)/size_piece1
        b1g2_piece1 = np.dot(b1_piece1, g2_piece1)/size_piece1
        b1b2_piece1 = np.dot(b1_piece1, b2_piece1)/size_piece1
        '''
        r1_piece2 = z[:, 0]-redav1_piece2
        r2_piece2 = z2[:, 0]-redav2_piece2
        g1_piece2 = z[:, 1]-greenav1_piece2
        g2_piece2 = z2[:, 1]-greenav2_piece2
        b1_piece2 = z[:, 2]-blueav1_piece2
        b2_piece2 = z2[:, 2]-blueav2_piece2
        '''
        size_piece2 = z2[:, 2].size
        size_piece2 = size_piece2/1.0
        r1r2_piece2 = np.dot(r1_piece2, r2_piece2)/size_piece2
        r1g2_piece2 = np.dot(r1_piece2, g2_piece2)/size_piece2
        r1b2_piece2 = np.dot(r1_piece2, b2_piece2)/size_piece2

        g1r2_piece2 = np.dot(g1_piece2, r2_piece2)/size_piece2
        g1g2_piece2 = np.dot(g1_piece2, g2_piece2)/size_piece2
        g1b2_piece2 = np.dot(g1_piece2, b2_piece2)/size_piece2

        b1r2_piece2 = np.dot(b1_piece2, r2_piece2)/size_piece2
        b1g2_piece2 = np.dot(b1_piece2, g2_piece2)/size_piece2
        b1b2_piece2 = np.dot(b1_piece2, b2_piece2)/size_piece2

        # this covarince matrix needs work  Test shuffling to see if scores are always the same
        covariance_piece1[0, 0] = r1r2_piece1  # top row
        covariance_piece1[0, 1] = (r1g2_piece1+g1r2_piece1)/2
        covariance_piece1[0, 2] = (r1b2_piece1+b1r2_piece1)/2

        covariance_piece1[1, 0] = (g1r2_piece1+r1g2_piece1)/2  # middle row
        covariance_piece1[1, 1] = g1g2_piece1
        covariance_piece1[1, 2] = (g1b2_piece1+b1g2_piece1)/2

        covariance_piece1[2, 0] = (b1r2_piece1+r1b2_piece1)/2
        covariance_piece1[2, 1] = (b1g2_piece1+g1b2_piece1)/2
        covariance_piece1[2, 2] = b1b2_piece1

        covariance_piece2[0, 0] = r1r2_piece2  # top row
        covariance_piece2[0, 1] = (r1g2_piece2+g1r2_piece2)/2
        covariance_piece2[0, 2] = (r1b2_piece2+b1r2_piece2)/2

        covariance_piece2[1, 0] = (g1r2_piece2+r1g2_piece2)/2  # middle row
        covariance_piece2[1, 1] = g1g2_piece2
        covariance_piece2[1, 2] = (g1b2_piece2+b1g2_piece2)/2

        covariance_piece2[2, 0] = (b1r2_piece2+r1b2_piece2)/2
        covariance_piece2[2, 1] = (b1g2_piece2+g1b2_piece2)/2
        covariance_piece2[2, 2] = b1b2_piece2
        '''
        red = r1_piece1-r1_piece2
        green = g1_piece1-g1_piece2
        blue = b1_piece1-b1_piece2
        redaverage = np.average(r1_piece1)
        greenaverage = np.average(g1_piece1)
        blueaverage = np.average(b1_piece1)

        # TODO use python cov not this crap
        #cov = np.linalg.pinv(covariance_piece1)
        # TODO do inverse!!!!!! Also need to use regular RGB!
        cov = np.linalg.pinv(sp.cov(a.T))

        red2 = r1_piece2-r1_piece1
        green2 = g1_piece2-g1_piece1
        blue2 = b1_piece2-b1_piece1
        redaverage2 = np.average(r1_piece2)
        greenaverage2 = np.average(g1_piece2)
        blueaverage2 = np.average(b1_piece2)
        #cov2 = np.linalg.pinv(covariance_piece2)

        cov2 = np.linalg.pinv(sp.cov(z.T))

        score = 0.0
        for i in range(len(red)):  # change this terrible way of doing it
            mymatrix = np.matrix(
                [red[i]-redaverage, green[i]-greenaverage, blue[i]-blueaverage])
            mymatrix2 = np.matrix(
                [red2[i]-redaverage2, green2[i]-greenaverage2, blue2[i]-blueaverage2])
            score += abs(mymatrix*cov*mymatrix.T)
            score += abs(mymatrix2*cov2*mymatrix2.T)
        return score

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
                        join_number] = self.mahalanobisDistance(self_top, self_top2, compare_bottom, compare_bottom2, join_number, "up")
        self.score_dict[own_number, JoinDirection.DOWN,
                        join_number] = self.mahalanobisDistance(self_bottom, self_bottom2, compare_top, compare_top2, join_number, "down")
        self.score_dict[own_number, JoinDirection.LEFT,
                        join_number] = self.mahalanobisDistance(self_left, self_left2, compare_right, compare_right2, join_number, "right")
        self.score_dict[own_number, JoinDirection.RIGHT,
                        join_number] = self.mahalanobisDistance(self_right, self_right2, compare_left, compare_left2, join_number, "left")

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
                   own_number] = euclideanDistance(self_top, compare_bottom)
        score_dict[join_number, JoinDirection.UP,
                   own_number] = euclideanDistance(self_bottom, compare_top)
        score_dict[join_number, JoinDirection.RIGHT,
                   own_number] = euclideanDistance(self_left, compare_right)
        score_dict[join_number, JoinDirection.LEFT,
                   own_number] = euclideanDistance(self_right, compare_left)


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
    score_dict={}
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[picX: picX+length, picY: picY+length, :]
            append(Segment(save, num_of_pieces_width,
                           num_of_pieces_height, piece_num, piece_num, score_dict))
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
        for segment2 in segment_list[index+1:]:
            if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
                segment1.calculateScoreEuclidean(segment2)
            if score_algorithum == ScoreAlgorithum.MAHALANOBIS:
                segment1.calculateScoreMahalonbis(segment2)


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


# no idea how to find the best one to start,  will do random for now! Myabe find piece with best connections
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


def saveImage(best_connection, peice_size, round, colortype):
    pic_locations = best_connection.binary_connection_matrix.nonzero()
    sizex = (max(pic_locations[0])-min(pic_locations[0]))+1
    sizey = (max(pic_locations[1])-min(pic_locations[1]))+1
    biggest_dim = sizex if sizex > sizey else sizey
    new_image = zeros((biggest_dim*peice_size, biggest_dim*peice_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matix[pic_locations[0]
                                                                 [x], pic_locations[1][x]].pic_matrix
        x1 = (pic_locations[0][x]-min(pic_locations[0]))*peice_size
        y1 = (pic_locations[1][x]-min(pic_locations[1]))*peice_size
        x2 = x1+peice_size
        y2 = y1+peice_size
        new_image[x1:x2, y1:y2, :] = piece_to_assemble
    if colortype == ColorType.LAB:
        new_image = color.lab2rgb(new_image)
    imageName = "round"+str(round)+".png"
    imsave(imageName, new_image)
    return imageName

# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?


def main():
    start_time = time.time()  # set up variables
    #parser = setUpArguments()
    picture_file_name = "william.png"  # parser.inputpic
    length = 60  # parser.length
    save_segments = True  # parser.savepieces
    image = imread(picture_file_name)  # parser.inputpic
    save_assembly_to_disk = True  # parser.saveassembly:
    show_building_animation = True  # parser.showanimation
    colorType = ColorType.LAB
    assemblyType = AssemblyType.PRIM
    scoreType = ScoreAlgorithum.EUCLIDEAN
    compareType = CompareWithOtherSegments.ONLY_BEST

    if colorType == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(image, length, save_segments, colorType)
    calculateScores(segment_list, scoreType)
    elapsed_time_secs = time.time() - start_time
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
        print("for round ", round, " i get score of ",
              best_connection.score, "the ratio for first to second best is ", best_connection.score/best_connection.second_best_score, " it took ", time.time()-start_time)
    elapsed_time_secs = time.time() - start_time
    print("Execution took: %s secs " % elapsed_time_secs)


if __name__ == '__main__':
    main()
