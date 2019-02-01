import argparse
import numpy as np
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

class Segment:
    pic_matrix=None
    score_dict={}
    binary_connection_matrix=np.asarray([[1,0],[0,0]])
    pic_connection_matix=None
    max_width = 0
    max_height = 0

    def __init__(self,pic_matrix, max_width, max_height):
        self.pic_matrix = pic_matrix
        self.pic_connection_matix = np.asarray([[pic_matrix,0], [0,0]])
        self.max_width = max_width
        self.max_height = max_height

    def calculateScore(self, segment, distanceMetric=DistanceMetric.EUCLIDEAN):
        self_top = self.pic_matrix[0:1, :, :]
        self_left = self.pic_matrix[:, 0:1, :]
        self_bottom = self.pic_matrix[self.pic_matrix.shape[0]-1:self.pic_matrix.shape[0], :, :]
        self_right = self.pic_matrix[:,self.pic_matrix.shape[0]-1:self.pic_matrix.shape[0], :]
        compare_top = segment.pic_matrix[0:1, :, :]
        compare_left = segment.pic_matrix[:, 0:1, :]
        compare_bottom = segment.pic_matrix[segment.pic_matrix.shape[0]-1:segment.pic_matrix.shape[0],: , :]
        compare_right = segment.pic_matrix[: ,segment.pic_matrix.shape[0]-1:segment.pic_matrix.shape[0], :]
        self.score_dict[JoinDirection.UP,segment] = np.linalg.norm(self_top-compare_bottom)
        self.score_dict[JoinDirection.DOWN, segment] = np.linalg.norm(self_bottom-compare_top)
        self.score_dict[JoinDirection.LEFT, segment] = np.linalg.norm(self_left - compare_right)
        self.score_dict[JoinDirection.RIGHT, segment] = np.linalg.norm(self_right-compare_left) 

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
    
    def getscore(self, pair1, pair2,  nodearray1, nodearray2, direction, compare_segment, r, c, location, currentround):
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
        stuff=temp.nonzero()
        for y in range(0, len(stuff[0])):
            storeing=stuff[0][y],stuff[1][y]
            distancex=storeing[0]-pair2[0]
            distancey=storeing[1]-pair2[1]
            padded1[pair2[0]+distancex][pair2[1]+distancey]=temp[storeing[0], storeing[1]]
        if temp[pair2[0], pair2[1]]==0 or padded1[pair2[0], pair2[1]]==0:
            raise ValueError('SOMETHING WENT WRONG')
        if direction=="right":
            self.bestpicarraytemp=padded1
            self.globaldirection="right"
            self.globalnode=node2
            return self.score_dict[JoinDirection.RIGHT,temp[pair2[0], pair2[1]].pic)]
        elif direction=="left":
            self.globaldirection="left"
            self.globalnode=node2
            self.bestpicarraytemp=padded1
            return self.scoreforleft(padded1[pair1[0], pair1[1]].pic, temp[pair2[0], pair2[1]].pic,node2, temp[pair2[0], pair2[1]], padded1[pair1[0], pair1[1]], currentround)
        elif direction=="down":
            self.globaldirection="down"
            self.globalnode=node2
            self.bestpicarraytemp=padded1
            return self.scoreforbottom(padded1[pair1[0], pair1[1]].pic, temp[pair2[0], pair2[1]].pic,node2, temp[pair2[0], pair2[1]], padded1[pair1[0], pair1[1]], currentround)
        elif direction=="up":
            self.globaldirection="up"
            self.globalnode=node2
            self.bestpicarraytemp=padded1
            return self.scorefortop(padded1[pair1[0], pair1[1]].pic, temp[pair2[0], pair2[1]].pic,node2,temp[pair2[0], pair2[1]], padded1[pair1[0], pair1[1]], currentround)


    def calculateConnections(self, compare_segment):
        h1 = self.binary_connection_matrix.shape[0]
        w1 = self.binary_connection_matrix.shape[1]
        h2 = compare_segment.binary_connection_matrix.shape[0]
        w2 = compare_segment.binary_connection_matrix.shape[1]
        pad_with_piece1 = np.zeros((h1+2*h2, w1+2*w2))
        pad_with_piece1[h2:(h2+h1),w2:(w2+w1)] = self.binary_connection_matrix
        dilation_mask = np.asarray([[0,1,0], [1,1,1,], [0,1,0]])
        result = binary_dilation(input=pad_with_piece1,structure=dilation_mask)
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
                        if pad_with_piece2[temp[0]][temp[1]+1]==1:
                           numofcompar+=1
                           score += self.score_dict[JoinDirection.LEFT,compare_segment.pic_connection_matix[temp[0], temp[1]]]
                        if pad_with_piece2[temp[0]][temp[1]-1]==1:
                           numofcompar+=1
                           score += self.score_dict[JoinDirection.RIGHT,compare_segment.pic_connection_matix[temp[0], temp[1]]]
                        if pad_with_piece2[temp[0]+1][temp[1]]==1:
                           numofcompar+=1
                           #score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),pad_with_piece1,pad_with_piece2,"up",node2, x, y,i,currentround))#down of the first one
                        if pad_with_piece2[temp[0]-1][temp[1]]==1:
                           numofcompar+=1
                           #score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),pad_with_piece1,pad_with_piece2,"down",node2, x,y, i,currentround)) #up of the first one       
    

def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pic", action="store", help="add picture you want to run on", required = True)
    parser.add_argument("-s", "--save", action="store_true", help="do you want the broken up peices save", default=False)
    parser.add_argument("-n", "--number", action="store", type=int, help="size of square segments wanted",required = True)
    return parser.parse_args() 

def breakUpImage(image,length,save_segments):
    image = misc.imread(image)
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
    for x in range(int(dimensions[0]/length)):
        for y in range(int(dimensions[1]/length)):
            save = image[picX: picX+length, picY: picY+length, :]
            segments.append(Segment(save,dimensions[0]/length, dimensions[1]/length))
            if save_segments:
                misc.imsave(str(x)+"_"+str(y)+".png", save)
            picY += length
        picX += length
        picY = 0
    return segments   

def calculateScores(segment_list):
    for segment1 in segment_list:
        for segment2 in segment_list:
            segment1.calculateScore(segment2)

def main():
    parser = setUpArguments()
    segment_list = breakUpImage(parser.pic, parser.number,parser.save)
    calculateScores(segment_list)
    for segment1 in segment_list:
        for segment2 in segment_list:
            segment1.calculateConnections(segment2)
    print("I STILL WORK")


if __name__ == '__main__':
    main()