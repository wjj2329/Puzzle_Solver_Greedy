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
from numpy import logical_and, zeros, nonzero, argwhere, delete, asarray, empty
from numpy import sum as numpySum
from numpy import all as numpyAll
from numpy.linalg import norm
import subprocess
import scipy.signal


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
    GIST_AND_EUCLDEAN = 3


class ColorType(Enum):
    RGB = 1
    LAB = 2


class AssemblyType(Enum):
    KRUSKAL = 1
    PRIM = 2


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

    def clear(self):
        self.pic_connection_matix = None
        self.join_segment = None
        self.score = sys.maxsize
        self.own_segment = None
        self.binary_connection_matrix = None
        self.second_best_score = sys.maxsize

    def isBetterConnection(self, otherConnection, compare_type):
        if compare_type == CompareWithOtherSegments.ONLY_BEST:
            return self.score < otherConnection.score
        # not this but something to use this ratio with the score
        elif compare_type == CompareWithOtherSegments.COMPARE_WITH_SECOND:
            return (2*(self.score*((self.score/self.second_best_score))))+self.score < otherConnection.score+(2*(otherConnection.score*((otherConnection.score/otherConnection.second_best_score))))

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

    def __eq__(self, other):
        return self.score == other.score


# malhnobis and different color spaces to try out.

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
        a = a.astype(np.int16)  # underflows would occur without this
        b = b.astype(np.int16)
        temp = [np.linalg.norm(x - y) for x, y in zip(a, b)]
        return sum(temp)

    def gistDistance(self, a, b, segment):
        colorScore = self.euclideanDistance(a, b)
        gistScore = self.euclideanDistance(
            np.asarray([self.gist]), np.asarray([segment.gist]))
        return (colorScore, gistScore)

    def mahalanobisDistance(self, a, a2, z, z2):
        a = a.astype(np.int16)  # underflows would occur without this
        a2 = a2.astype(np.int16)
        z = z.astype(np.int16)
        z2 = z2.astype(np.int16)

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


# http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
# https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
# https://www.python.org/dev/peps/pep-0371/ use this to make it faster
# https://www.sciencedirect.com/science/article/pii/S131915781830394X gist combo with euclidean
# https://pdfs.semanticscholar.org/4003/7d131e3365feb9d69912b3c8e8527e9ed2d5.pdf  cycle detection
# Filter the image?  Gausian blur etc?


    def calculateScoreMahalonbis(self, segment):
        pic_matrix = self.pic_matrix
        mahalanobisDistance = self.mahalanobisDistance
        score_dict = self.score_dict
        self_top = pic_matrix[0, :, :]
        self_top2 = pic_matrix[1, :, :]
        self_left = pic_matrix[:, 0, :]
        self_left2 = pic_matrix[:, 1, :]
        self_bottom = pic_matrix[-1, :, :]
        self_bottom2 = pic_matrix[-2, :, :]
        self_right = pic_matrix[:, -1, :]
        self_right2 = pic_matrix[:, -2, :]

        segment_matrix = segment.pic_matrix
        compare_top = segment_matrix[0, :, :]
        compare_top2 = segment_matrix[1, :, :]
        compare_left = segment_matrix[:, 0, :]
        compare_left2 = segment_matrix[:, 1, :]
        compare_bottom = segment_matrix[-1, :, :]
        compare_bottom2 = segment_matrix[-2, :, :]
        compare_right = segment_matrix[:, -1, :]
        compare_right2 = segment_matrix[:, -2, :]

        own_number = self.piece_number
        join_number = segment.piece_number
        score_dict[own_number, JoinDirection.UP,
                   join_number] = mahalanobisDistance(self_top, self_top2, compare_bottom, compare_bottom2)
        score_dict[own_number, JoinDirection.DOWN,
                   join_number] = mahalanobisDistance(self_bottom, self_bottom2, compare_top, compare_top2)
        score_dict[own_number, JoinDirection.LEFT,
                   join_number] = mahalanobisDistance(self_left, self_left2, compare_right, compare_right2)
        score_dict[own_number, JoinDirection.RIGHT,
                   join_number] = mahalanobisDistance(self_right, self_right2, compare_left, compare_left2)

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

    def calculateScoreGIST(self, segment): #this doesn't work :(
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
        score_dict = self.score_dict
        euclideanDistance = self.euclideanDistance

        pic_matrix = self.pic_matrix
        self_top = pic_matrix[0, :, :]
        self_left = pic_matrix[:, 0, :]
        self_bottom = pic_matrix[-1, :, :]
        self_right = pic_matrix[:, -1, :]

        segment_matrix = segment.pic_matrix
        compare_top = segment_matrix[0, :, :]
        compare_left = segment_matrix[:, 0, :]
        compare_bottom = segment_matrix[-1, :, :]
        compare_right = segment_matrix[:, -1, :]

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
    def checkforcompatibility(self, booleanarray, max_height, max_width):
        non_zero_values = nonzero(booleanarray)
        smallestx1 = min(non_zero_values[1])
        smallesty1 = min(non_zero_values[0])
        biggestx1 = max(non_zero_values[1])
        biggesty1 = max(non_zero_values[0])
        if biggestx1-smallestx1 > max_height or biggesty1-smallesty1 > max_width:
            return False
        return True

    # TODO what about a combination of both kruskal and prims like divide into quatars prims?
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
        # findValuesToCompare=self.findValuesToCompare
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
                    temp_pointer = zeros(  # just use fill to speed this up?
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
                    imsave(str(x)+"_"+str(y)+".png", save)
                elif colortype == ColorType.LAB:
                    imageTemp = color.lab2rgb(save)
                    imsave(str(x)+"_"+str(y)+".png", imageTemp)
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


def calculateScores(segment_list, score_algorithum):
    for index, segment1 in enumerate(segment_list):
        print("calcuating score for segment ", segment1.piece_number)
        for segment2 in segment_list[index+1:]:
            if score_algorithum == ScoreAlgorithum.EUCLIDEAN:
                segment1.calculateScoreEuclidean(segment2)
            elif score_algorithum == ScoreAlgorithum.MAHALANOBIS:
                segment1.calculateScoreMahalonbis(segment2)
            elif score_algorithum == ScoreAlgorithum.GIST_AND_EUCLDEAN:
                segment1.calculateScoreGIST(segment2)


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
        for key in dict(connection.connections_dict):
            if key[0] == removing or key[1] == removing:
                del connection.connections_dict[key]


def saveImage(best_connection, piece_size, round, colortype, name_for_round):
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
    imageName = name_for_round+" round"+str(round)+".png"
    imsave(imageName, new_image)
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


# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?
# TODO Maybe have it go in lines? Or at least start off with two lines one horizontal one vertical to build off and stop going out of bounds?
# TODO maybe combo of kruskal and prims? Divide into blocks? LImit the number of trees? Force to use prims after awhile?

def main():  # TODO SOMETHING IS MAKING THIS GET DIFFERENT ASSEMBLY RESULTS THIS IS A PROBLEM!
    start_time = time.time()
    picture_file_name = "william.png"
    length = 30
    save_segments = True
    image = imread(picture_file_name)
    save_assembly_to_disk = True
    show_building_animation = True
    show_print_statements = True
    boost_priority_of_big_pieces_joining = False
    use_cross = False
    colorType = ColorType.LAB
    assemblyType = AssemblyType.KRUSKAL
    scoreType = ScoreAlgorithum.MAHALANOBIS
    compareType = CompareWithOtherSegments.ONLY_BEST
    name_for_round = "test"

    if colorType == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(
        image, length, save_segments, colorType, scoreType)
    calculateScores(segment_list, scoreType)
    normalizeScores(segment_list, scoreType)
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
                segment_list, compareType, boost_priority_of_big_pieces_joining, compareType)
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
                Image.open(saveImage(best_connection, length, round, colorType, name_for_round)))
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
