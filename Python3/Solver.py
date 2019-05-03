import argparse
import numpy as np
import random
import sys
from imageio import imread, imsave
from enum import Enum
from scipy.ndimage.morphology import binary_dilation
import tkinter
from PIL import ImageTk, Image
from skimage import io, color


class JoinDirection(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class BestConnection:
    pic_connection_matix = None
    join_segment = None
    score = sys.maxsize
    own_segment = None
    binary_connection_matrix = None

    def __init__(self, own_segment=None, pic_connection_matix=None, join_segment=None, binary_connection_matrix=None):
        self.own_segment = own_segment
        self.pic_connection_matix = pic_connection_matix
        self.join_segment = join_segment
        self.binary_connection_matrix = binary_connection_matrix

    def isBetterConnection(self, otherConnection):
        return self.score < otherConnection.score

    def setNodeContents(self, my_list):
        my_list.remove(self.own_segment)
        self.own_segment.pic_connection_matix = self.pic_connection_matix
        self.own_segment.binary_connection_matrix = self.binary_connection_matrix
        my_list.append(self.own_segment)

    def stripZeros(self):
        self.pic_connection_matix = self.pic_connection_matix[~np.all(
            self.pic_connection_matix == 0, axis=1)]
        self.binary_connection_matrix = self.binary_connection_matrix[~np.all(
            self.binary_connection_matrix == 0, axis=1)]
        idx = np.argwhere(
            np.all(self.pic_connection_matix[..., :] == 0, axis=0))
        self.pic_connection_matix = np.delete(
            self.pic_connection_matix, idx, axis=1)
        idx = np.argwhere(
            np.all(self.binary_connection_matrix[..., :] == 0, axis=0))
        self.binary_connection_matrix = np.delete(
            self.binary_connection_matrix, idx, axis=1)


# malhnobis and different color spaces to try out.

class Segment:
    myownNumber = 0
    pic_matrix = None
    score_dict = {}
    connections_dict = {}
    binary_connection_matrix = np.asarray([[1, 0], [0, 0]])
    pic_connection_matix = None
    max_width = 0
    max_height = 0
    best_connection_found_so_far = BestConnection()
    piece_number = -1

    def __init__(self, pic_matrix, max_width, max_height, piece_number, myownNumber):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = np.asarray([[self, 0], [0, 0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number
        self.myownNumber = myownNumber

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
        covariance_piece1 = np.zeros([3, 3])
        covariance_piece2 = np.zeros([3, 3])

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

        r1_piece2 = z[:, 0]-redav1_piece2
        r2_piece2 = z2[:, 0]-redav2_piece2
        g1_piece2 = z[:, 1]-greenav1_piece2
        g2_piece2 = z2[:, 1]-greenav2_piece2
        b1_piece2 = z[:, 2]-blueav1_piece2
        b2_piece2 = z2[:, 2]-blueav2_piece2
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

        red = r1_piece1-r1_piece2
        green = g1_piece1-g1_piece2
        blue = b1_piece1-b1_piece2
        redaverage = np.average(r1_piece1)
        greenaverage = np.average(g1_piece1)
        blueaverage = np.average(b1_piece1)
        cov = np.linalg.pinv(covariance_piece1)

        try:
            something = np.linalg.inv(covariance_piece1)
        except:
            print("my matrix ", covariance_piece1, " this piece ", self.piece_number,
                  " with piece ", piece2Num, " direction ", direction, " cov")
        try:
            something2 = np.linalg.inv(covariance_piece2)
        except:
            print("my matrix ", covariance_piece2, " this piece ", self.piece_number,
                  " with piece ", piece2Num, " direction ", direction, " cov2")

        red2 = r1_piece2-r1_piece1
        green2 = g1_piece2-g1_piece1
        blue2 = b1_piece2-b1_piece1
        redaverage2 = np.average(r1_piece2)
        greenaverage2 = np.average(g1_piece2)
        blueaverage2 = np.average(b1_piece2)
        cov2 = np.linalg.pinv(covariance_piece2)

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
        self.score_dict[own_number, JoinDirection.UP,
                        join_number] = self.euclideanDistance(self_top, compare_bottom)
        self.score_dict[own_number, JoinDirection.DOWN,
                        join_number] = self.euclideanDistance(self_bottom, compare_top)
        self.score_dict[own_number, JoinDirection.LEFT,
                        join_number] = self.euclideanDistance(self_left, compare_right)
        self.score_dict[own_number, JoinDirection.RIGHT,
                        join_number] = self.euclideanDistance(self_right, compare_left)

    def checkforcompatibility(self, booleanarray):
        whattokeep = np.nonzero(booleanarray)
        smallestx1 = min(np.nonzero(booleanarray)[1])
        smallesty1 = min(np.nonzero(booleanarray)[0])
        biggestx1 = max(np.nonzero(booleanarray)[1])
        biggesty1 = max(np.nonzero(booleanarray)[0])
        biggest = biggestx1-smallestx1+1
        if(biggesty1 - smallesty1 + 1 > biggest):
            biggest = biggesty1 - smallesty1 + 1
        storeing = np.zeros((biggest, biggest), dtype="object")
        for y in range(0, len(whattokeep[0])):
            pair = [whattokeep[0][y], whattokeep[1][y]]
        storeing[pair[0] - smallesty1][pair[1] -
                                       smallestx1] = booleanarray[pair[0]][pair[1]]
        temp = storeing
        if temp.shape[0] > self.max_height or temp.shape[1] > self.max_width:
            return False
        return True

    def printPictureNumberMatrix(self, matrix):
        matrix = matrix[~np.all(matrix == 0, axis=1)]
        idx = np.argwhere(np.all(matrix[..., :] == 0, axis=0))
        matrix = np.delete(matrix, idx, axis=1)

        for row in matrix:
            for val in row:
                if val == 0:
                    print(val, end=" ")
                else:
                    print(val.piece_number, end=" ")
            print()

    def calculateConnections(self, compare_segment, round):
        if (self.myownNumber, compare_segment.myownNumber) in self.connections_dict:
            return self.connections_dict[(self.myownNumber, compare_segment.myownNumber)]
        h1 = self.binary_connection_matrix.shape[0]
        w1 = self.binary_connection_matrix.shape[1]
        h2 = compare_segment.binary_connection_matrix.shape[0]
        w2 = compare_segment.binary_connection_matrix.shape[1]
        pad_with_piece1 = np.zeros((h1+2*h2, w1+2*w2))
        pad_with_piece1[h2:(h2+h1), w2:(w2+w1)] = self.binary_connection_matrix
        dilation_mask = np.asarray([[0, 1, 0], [1, 1, 1, ], [0, 1, 0]])
        result = binary_dilation(
            input=pad_with_piece1, structure=dilation_mask)
        neighboring_connections = result - pad_with_piece1
        for x in range(h1+2*h2-(h2-1)):
            for y in range(w1+2*w2-(w2-1)):
                pad_with_piece2 = np.zeros(neighboring_connections.shape)
                pad_with_piece2[x:(x+h2), y:(y+w2)
                                ] = compare_segment.binary_connection_matrix
                connect_map = np.logical_and(
                    neighboring_connections, pad_with_piece2)
                overlap_map = np.logical_and(pad_with_piece1, pad_with_piece2)
                has_connections = np.sum(connect_map[:]) > 0
                has_overlap = np.sum(overlap_map[:]) > 0
                combined_pieces = pad_with_piece1+pad_with_piece2
                if has_connections and not has_overlap and self.checkforcompatibility(combined_pieces):
                    store = np.nonzero(pad_with_piece1)
                    score = 0
                    numofcompar = 0
                    padded1_pointer = np.zeros(
                        (h1+2*h2, w1+2*w2), dtype="object")
                    padded1_pointer[h2:(h2+h1), w2:(w2+w1)
                                    ] = self.pic_connection_matix
                    temp_pointer = np.zeros((h1+2*h2, w1+2*w2), dtype="object")
                    temp_pointer[x:(h2+x), y:(w2+y)
                                 ] = compare_segment.pic_connection_matix
                    distancex = 0
                    distancey = 0
                    stuff = temp_pointer.nonzero()
                    pair2 = (connect_map.nonzero()[
                             0][0], connect_map.nonzero()[1][0])
                    for y in range(0, len(stuff[0])):
                        storeing = stuff[0][y], stuff[1][y]
                        distancex = storeing[0]-pair2[0]
                        distancey = storeing[1]-pair2[1]
                        first = pair2[0]+distancex
                        second = pair2[1]+distancey
                        padded1_pointer[first][second] = temp_pointer[storeing[0], storeing[1]]
                    for i in range(0, len(store[0])):
                        temp = [store[0][i], store[1][i]]
                        # piece two, direction
                        if pad_with_piece2[temp[0]][temp[1]+1] == 1:
                            node1 = padded1_pointer[temp[0], temp[1]]
                            node2 = padded1_pointer[temp[0], temp[1]+1]
                            numofcompar += 1
                            score += self.score_dict[node1.piece_number,
                                                     JoinDirection.RIGHT, node2.piece_number]
                        if pad_with_piece2[temp[0]][temp[1]-1] == 1:
                            node1 = padded1_pointer[temp[0], temp[1]]
                            node2 = padded1_pointer[temp[0], temp[1]-1]
                            numofcompar += 1
                            score += self.score_dict[node1.piece_number,
                                                     JoinDirection.LEFT, node2.piece_number]
                        if pad_with_piece2[temp[0]+1][temp[1]] == 1:
                            node1 = padded1_pointer[temp[0], temp[1]]
                            node2 = padded1_pointer[temp[0]+1, temp[1]]
                            numofcompar += 1
                            score += self.score_dict[node1.piece_number,
                                                     JoinDirection.DOWN, node2.piece_number]
                        if pad_with_piece2[temp[0]-1][temp[1]] == 1:
                            node1 = padded1_pointer[temp[0], temp[1]]
                            node2 = padded1_pointer[temp[0]-1, temp[1]]
                            numofcompar += 1
                            score += self.score_dict[node1.piece_number,
                                                     JoinDirection.UP, node2.piece_number]
                    score = score/numofcompar
                    if score < self.best_connection_found_so_far.score:
                        self.best_connection_found_so_far.pic_connection_matix = padded1_pointer
                        self.best_connection_found_so_far.binary_connection_matrix = combined_pieces
                        self.best_connection_found_so_far.score = score
                        self.best_connection_found_so_far.own_segment = self
                        self.best_connection_found_so_far.join_segment = compare_segment
        self.connections_dict[(
            self.myownNumber, compare_segment.myownNumber)] = self.best_connection_found_so_far
        return self.best_connection_found_so_far


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
    return parser.parse_args()


def breakUpImage(image, length, save_segments, cielab):
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
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[picX: picX+length, picY: picY+length, :]
            append(Segment(save, num_of_pieces_width,
                           num_of_pieces_height, piece_num, piece_num))
            piece_num += 1
            if save_segments:
                imsave(str(x)+"_"+str(y)+".png", save)
            picY += length
        picX += length
        picY = 0
    return segments


def calculateScores(segment_list):
    for segment1 in segment_list:
        for segment2 in segment_list:
            if segment1 != segment2:
                segment1.calculateScoreEuclidean(segment2)


# TODO make this a map instead of doing the same comparison over and over again.
def findBestConnection(segment_list, round):
    best_so_far = BestConnection()
    for index, segment1 in enumerate(segment_list):
        for segment2 in segment_list[index+1:]:
            segment1.best_connection_found_so_far=BestConnection()
            temp = segment1.calculateConnections(segment2, round)
            segment1.best_connection_found_so_far=BestConnection()
            #print(segment1.myownNumber, segment2.myownNumber)
            if temp.isBetterConnection(best_so_far):
                best_so_far = temp
    return best_so_far


def printPiecesMatrices(segment_list):
    for node in segment_list:
        print(node.binary_connection_matrix)
        print(node.pic_connection_matix)
        print(node.piece_number)
    print('\n\n\n')


def resetConnection(my_list, removing):
    for connection in my_list:
        connection.best_connection_found_so_far = BestConnection()
        '''
        newscoredict={}
        for key,val in connection.connections_dict:
            if key !=removing and val!=removing:
                newscoredict[(key,val)]=connection.connections_dict[(key,val)]
        connection.connections_dict=newscoredict
        '''        


def saveImage(best_connection, peice_size, round, cielab):
    pic_locations = best_connection.binary_connection_matrix.nonzero()
    sizex = (max(pic_locations[0])-min(pic_locations[0]))+1
    sizey = (max(pic_locations[1])-min(pic_locations[1]))+1
    biggest_dim = sizex if sizex > sizey else sizey
    new_image = np.zeros((biggest_dim*peice_size, biggest_dim*peice_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matix[pic_locations[0]
                                                                 [x], pic_locations[1][x]].pic_matrix
        x1 = (pic_locations[0][x]-min(pic_locations[0]))*peice_size
        y1 = (pic_locations[1][x]-min(pic_locations[1]))*peice_size
        x2 = x1+peice_size
        y2 = y1+peice_size
        new_image[x1:x2, y1:y2, :] = piece_to_assemble
    #print("image before is ", new_image)
    if cielab:
        new_image = color.lab2rgb(new_image)
        #print("my new image is ",new_image)
    imageName = "roundTest"+str(round)+".png"
    imsave(imageName, new_image)
    return imageName

#todo do it with prims and KRUSKALS both ways
def main():

    #parser = setUpArguments()
    cielab = True  # parser.cielab
    image = imread("william.png")  # parser.inputpic
    if(cielab):
        image = io.imread("william.png")
        image = color.rgb2lab(image)
    # parser.length,parser.savepieces)
    segment_list = breakUpImage(image, 30, True, True)
    calculateScores(segment_list)
    window = tkinter.Tk()
    window.title("Picture")
    img = ImageTk.PhotoImage(Image.open("william.png"))
    w = tkinter.Label(window, image=img)
    random.shuffle(segment_list)
    round = 0
    original_size = len(segment_list)
    while len(segment_list) > 1:
        best_connection = findBestConnection(segment_list, round)
        best_connection.stripZeros()
        best_connection.own_segment.binary_connection_matrix = best_connection.binary_connection_matrix
        best_connection.own_segment.pic_connection_matix = best_connection.pic_connection_matix
        origional=best_connection.own_segment.myownNumber
        best_connection.own_segment.myownNumber += original_size
        segment_list.remove(best_connection.join_segment)
        if True:  # parser.saveassembly:
            updated_picture = ImageTk.PhotoImage(
                Image.open(saveImage(best_connection, 30, round, cielab)))
            w.configure(image=updated_picture)
            w.image = updated_picture
            w.pack(side="bottom", fill="both", expand="no")
            window.update()
        round += 1
        print("for round ", round, " i get score of ", best_connection.score)
        #print("i should remove ", origional, best_connection.join_segment.myownNumber)
        resetConnection(segment_list, origional)
        #resetConnection(segment_list, best_connection.join_segment.myownNumber)


if __name__ == '__main__':
    main()
