import argparse
import numpy as np
import random
import sys
from scipy import misc
from enum import Enum
from scipy.ndimage.morphology import binary_dilation

class JoinDirection(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class DistanceMetric(Enum):
    EUCLIDEAN = 1
    MAHALANOBIS = 2  

class BestConnection:
    pic_connection_matix = None
    join_direction = None
    join_segment = None
    score = sys.maxsize
    own_segment = None
    binary_connection_matrix = None

    def __init__(self, pic_connection_matix=None, join_direction=None, join_segment=None, binary_connection_matrix=None):
        self.join_direction = join_direction
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


class Segment:
    pic_matrix=None
    score_dict={}
    binary_connection_matrix=np.asarray([[1,0],[0,0]])
    pic_connection_matix=None
    max_width = 0
    max_height = 0
    best_connection_found_so_far = BestConnection()
    connection_to_compare = BestConnection()
    piece_number =-1

    def __init__(self,pic_matrix, max_width, max_height, piece_number):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = np.asarray([[self,0], [0,0]])
        self.max_width = max_width
        self.max_height = max_height
        self.piece_number = piece_number

    def euclideanDistance(self, a, b):
        a=a[0].astype(np.int16) #underflows would occur without this
        b=b[0].astype(np.int16)
        temp = [np.linalg.norm(x[0]-y[0])+np.linalg.norm(x[1]-y[1])+np.linalg.norm(x[2]-y[2]) for x, y in zip(a, b)]
        return sum(temp)

    def calculateScore(self, segment, distanceMetric=DistanceMetric.EUCLIDEAN):
        rot = np.rot90(self.pic_matrix)
        self_top = self.pic_matrix[0:1, :, :]
        self_left = rot[self.pic_matrix.shape[0]-1:self.pic_matrix.shape[0], :, :]
        self_bottom = self.pic_matrix[self.pic_matrix.shape[0]-1:self.pic_matrix.shape[0], :, :]
        self_right = rot[0:1, :, :]

        rot=np.rot90(segment.pic_matrix)
        compare_top = segment.pic_matrix[0:1, :, :]
        compare_left = rot[segment.pic_matrix.shape[0]-1:segment.pic_matrix.shape[0],:, :]
        compare_bottom = segment.pic_matrix[segment.pic_matrix.shape[0]-1:segment.pic_matrix.shape[0],: , :]
        compare_right =rot[0:1, :, :] 
        #print("I place up for segment ", self.piece_number, " with ", segment.piece_number, " with score ",self.euclideanDistance(self_top,compare_bottom) )        
        #print("I place down for segment ", self.piece_number, " with ", segment.piece_number, " with score ",self.euclideanDistance(self_bottom,compare_top) )        
        #print("I place left for segment ", self.piece_number, " with ", segment.piece_number, " with score ",self.euclideanDistance(self_left , compare_right) )        
        #print("I place right for segment ", self.piece_number, " with ", segment.piece_number, " with score ",self.euclideanDistance(self_right,compare_left) )          
        self.score_dict[segment.piece_number,JoinDirection.UP,self.piece_number] = self.euclideanDistance(self_top,compare_bottom)
        self.score_dict[segment.piece_number,JoinDirection.DOWN, self.piece_number] = self.euclideanDistance(self_bottom,compare_top)
        self.score_dict[segment.piece_number,JoinDirection.LEFT, self.piece_number] = self.euclideanDistance(self_left , compare_right)
        self.score_dict[segment.piece_number,JoinDirection.RIGHT, self.piece_number] = self.euclideanDistance(self_right,compare_left) 

    def checkforcompatibility(self, booleanarray):
        whattokeep=np.nonzero(booleanarray)
        smallestx1=min(np.nonzero(booleanarray)[1])
        smallesty1=min(np.nonzero(booleanarray)[0])
        biggestx1=max(np.nonzero(booleanarray)[1])
        biggesty1=max(np.nonzero(booleanarray)[0])
        biggest=biggestx1-smallestx1+1
        if(biggesty1-smallesty1+1>biggest):
            biggest=biggesty1-smallesty1+1
        storeing=np.zeros((biggest,biggest ), dtype="object")
        for y in range(0, len(whattokeep[0])):
            pair=[whattokeep[0][y], whattokeep[1][y]]
        storeing[pair[0]-smallesty1] [pair[1]-smallestx1]=booleanarray[pair[0]] [ pair[1]]
        temp=storeing
        if temp.shape[0]>self.max_height or temp.shape[1]>self.max_width:
            return False
        return True
    
    def getscore(self, pair2, direction, compare_segment, r, c, binary_matrix):
        piece1 = self.pic_connection_matix
        piece2 = compare_segment.pic_connection_matix
        h1 = piece1.shape[0]
        w1 = piece1.shape[1]
        h2 = piece2.shape[0]
        w2 = piece2.shape[1]
        oldpadded1 = np.zeros( (h1+2*h2,w1+2*w2), dtype="object" )
        padded1 = np.zeros( (h1+2*h2,w1+2*w2), dtype="object" )
        padded1[h2:(h2+h1),w2:(w2+w1)] = piece1
        temp = np.zeros( (h1+2*h2,w1+2*w2), dtype="object"  )
        temp[r:(h2+r),c:(w2+c)] = piece2
        distancex = 0
        distancey = 0
        stuff = temp.nonzero()
        for y in range(0, len(stuff[0])):
            storeing = stuff[0][y],stuff[1][y]
            distancex = storeing[0]-pair2[0]
            distancey = storeing[1]-pair2[1]
            padded1[pair2[0]+distancex][pair2[1]+distancey] = temp[storeing[0], storeing[1]]
        if temp[pair2[0], pair2[1]] == 0 or padded1[pair2[0], pair2[1]] == 0:
            raise ValueError('SOMETHING WENT WRONG')
        #print("I am ", self.piece_number," ",temp[pair2[0], pair2[1]].piece_number)
        if direction == JoinDirection.RIGHT:
            self.best_connection_to_compare=BestConnection(padded1, JoinDirection.LEFT, compare_segment, binary_matrix)
            return self.score_dict[self.piece_number,JoinDirection.RIGHT,temp[pair2[0], pair2[1]].piece_number]
        elif direction == JoinDirection.LEFT:
            self.best_connection_to_compare=BestConnection(padded1, JoinDirection.RIGHT, compare_segment, binary_matrix)
            return self.score_dict[self.piece_number,JoinDirection.LEFT, temp[pair2[0], pair2[1]].piece_number]
        elif direction == JoinDirection.DOWN:
            self.best_connection_to_compare=BestConnection(padded1, JoinDirection.UP, compare_segment, binary_matrix)
            return self.score_dict[self.piece_number,JoinDirection.DOWN, temp[pair2[0], pair2[1]].piece_number]
        elif direction == JoinDirection.UP:
            self.best_connection_to_compare=BestConnection(padded1, JoinDirection.DOWN, compare_segment, binary_matrix)
            return self.score_dict[self.piece_number,JoinDirection.UP, temp[pair2[0], pair2[1]].piece_number]
    def printPictureNumberMatrix(self,matrix):
        for row in matrix:
             for val in row:
                if val == 0:    
                    print (val,end=" ")
                else:
                     print(val.piece_number,end=" ")   
             print()

 
    def calculateConnections(self, compare_segment, round):
        h1 = self.binary_connection_matrix.shape[0]
        w1 = self.binary_connection_matrix.shape[1]
        h2 = compare_segment.binary_connection_matrix.shape[0]
        w2 = compare_segment.binary_connection_matrix.shape[1]
        pad_with_piece1 = np.zeros((h1+2*h2, w1+2*w2))
        pad_with_piece1[h2:(h2+h1),w2:(w2+w1)] = self.binary_connection_matrix
        dilation_mask = np.asarray([[0,1,0], [1,1,1,], [0,1,0]])
        result = binary_dilation(input = pad_with_piece1,structure=dilation_mask)
        neighboring_connections = result - pad_with_piece1
        for x in range(h1+2*h2-(h2-1)):
            for y in range(w1+2*w2-(w2-1)):
                pad_with_piece2 = np.zeros(neighboring_connections.shape)
                pad_with_piece2[x:(x+h2),y:(y+w2)] = compare_segment.binary_connection_matrix
                connect_map = np.logical_and(neighboring_connections,pad_with_piece2)
                overlap_map = np.logical_and(pad_with_piece1,pad_with_piece2)
                has_connections = np.sum(connect_map[:]) > 0
                has_overlap = np.sum(overlap_map[:]) > 0
                combined_pieces = pad_with_piece1+pad_with_piece2
                if has_connections and not has_overlap and self.checkforcompatibility(combined_pieces):
                    combined_pieces = pad_with_piece1+pad_with_piece2
                    store = np.nonzero(pad_with_piece1) 
                    score = 0 
                    numofcompar = 0
                    for i in range(0,len(store[0])):
                        temp = [store[0][i], store[1][i]] 
                        if pad_with_piece2[temp[0]][temp[1]+1] == 1:
                           numofcompar+=1
                           score+=self.getscore( (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]), JoinDirection.LEFT, compare_segment, x, y, combined_pieces)#down of the first one
                        if pad_with_piece2[temp[0]][temp[1]-1] == 1:
                           numofcompar+=1
                           score+=self.getscore( (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]), JoinDirection.RIGHT, compare_segment, x, y, combined_pieces)#down of the first one
                        if pad_with_piece2[temp[0]+1][temp[1]] == 1:
                           numofcompar+=1
                           score+=self.getscore( (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]), JoinDirection.UP, compare_segment, x, y, combined_pieces)#down of the first one
                        if pad_with_piece2[temp[0]-1][temp[1]] == 1:
                           numofcompar+=1
                           score+=self.getscore( (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]), JoinDirection.DOWN, compare_segment, x,y, combined_pieces) #up of the first one       
                    score/=numofcompar
                    self.best_connection_to_compare.score=score
                    #print("i get score of ",score, "my file is called ", round)
                    #self.printPictureNumberMatrix(self.best_connection_to_compare.pic_connection_matix)
                    #saveImage(self.best_connection_to_compare,480, round)
                    #round+=100
                    if self.best_connection_to_compare.isBetterConnection(self.best_connection_found_so_far):
                        self.best_connection_found_so_far=self.best_connection_to_compare
        return self.best_connection_found_so_far                
                

def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputpic", action = "store", help="add picture you want to run on", required = True)
    parser.add_argument("-sp", "--savepieces", action = "store_true", help="save the pieces the picture was broken up into", default = False)
    parser.add_argument("-l", "--length", action = "store", type = int, help="size of the length of square segments wanted in pixels",required = True)
    parser.add_argument("-sa", "--saveassembly", action = "store_true", help="save the assembled picture in each round", default = False)
    return parser.parse_args() 

def breakUpImage(image,length,save_segments):
    dimensions = image.shape
    if dimensions[0] != dimensions[1]:
        print("Only square images will work for now to keep things simple")
        exit()
    if dimensions[0]%length != 0 or dimensions[1]%length != 0:
        print("unable to break up image into equal squares")
        exit()
    segments = []
    x, y = 0, 0
    picX, picY = 0, 0
    piece_num = 1
    for x in range(int(dimensions[0]/length)):
        for y in range(int(dimensions[1]/length)):
            save = image[picX: picX+length, picY: picY+length, :]
            segments.append(Segment(save,dimensions[0]/length, dimensions[1]/length, piece_num))
            piece_num+=1
            if save_segments:
                misc.imsave(str(x)+"_"+str(y)+".png", save)
            picY += length
        picX += length
        picY = 0
    return segments   

def calculateScores(segment_list):
    for segment1 in segment_list:
        for segment2 in segment_list:
            if segment1 != segment2:
                segment1.calculateScore(segment2)

def findBestConnection(segment_list):
    best_so_far=BestConnection()
    round=1
    for index, segment1 in enumerate(segment_list):
        for segment2 in segment_list[index+1:]:
             temp=segment1.calculateConnections(segment2,round)
             round+=1
             if temp.isBetterConnection(best_so_far):
                 best_so_far = temp
                 best_so_far.own_segment = segment1
                 best_so_far.join_segment = segment2
    print(" I found the best score of ",best_so_far.score)             
    return best_so_far   

def printPiecesMatrices(segment_list):
    for node in segment_list:
        print(node.binary_connection_matrix)
        print(node.pic_connection_matix)
    print('\n\n\n')

def resetConnection(my_list):
    for connection in my_list:
        connection.best_connection_found_so_far=BestConnection()
        connection.best_connection_to_compare=BestConnection()        

def saveImage(best_connection,peice_size, round):
    pic_locations=best_connection.binary_connection_matrix.nonzero()
    biggest=max(pic_locations[0]) if max(pic_locations[0])>max(pic_locations[1]) else max(pic_locations[1])
    smallest=min(pic_locations[0]) if min(pic_locations[0])<min(pic_locations[1])else min(pic_locations[1])
    sizex=(max(pic_locations[0])-min(pic_locations[0]))+1
    sizey=(max(pic_locations[1])-min(pic_locations[1]))+1    
    print("my size x is ", sizex, " my size y is ", sizey)
    biggest_dim= sizex if sizex>sizey else sizey
    new_image=np.zeros((biggest_dim*peice_size, biggest_dim*peice_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matix[pic_locations[0][x], pic_locations[1][x]].pic_matrix
        print(pic_locations)
        x1 = (pic_locations[0][x]-min(pic_locations[0]))*peice_size
        y1 = (pic_locations[1][x]-min(pic_locations[1]))*peice_size
        x2 = x1+peice_size
        y2 = y1+peice_size
        print("these are the dimensions ",x1, x2, y1, y2, new_image.shape)
        new_image[x1:x2, y1:y2, :] = piece_to_assemble
    misc.imsave("round"+str(round)+".png", new_image)
 

def main():
    parser = setUpArguments()
    image = misc.imread(parser.inputpic)
    segment_list = breakUpImage(image, parser.length,parser.savepieces)
    calculateScores(segment_list)
    #random.shuffle(segment_list)
    round = 1
    while len(segment_list)>1:
        best_connection=findBestConnection(segment_list)
        segment_list.remove(best_connection.join_segment)
        best_connection.setNodeContents(segment_list)
        if parser.saveassembly:
            saveImage(best_connection, parser.length, round)
        round += 1
        resetConnection(segment_list)
        #return

   

if __name__ == '__main__':
    main()