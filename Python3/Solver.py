import argparse
import numpy as np
from scipy import misc
from enum import Enum
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
     def __init__(self,pic_matrix):
         self.pic_matrix=pic_matrix
         
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
         return None


def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pic", action="store", help="add picture you want to run on", required = True)
    parser.add_argument("-s", "--save", action="store_true", help="do you want the broken up peices save", default=False)
    parser.add_argument("-n", "--number", action="store", type=int, help="size of square segments wanted",required = True)
    return parser.parse_args() 

def breakUpImage(image,length,save_segments):
    image=misc.imread(image)
    dimensions=image.shape
    if dimensions[0] != dimensions[1]:
        print("Only square images will work for now to keep things simple")
        exit()
    if dimensions[0]%length != 0 or dimensions[1]%length != 0:
        print("unable to break up image into equal squares")
        exit()
    segments = []
    x, y = 0, 0
    picX,picY = 0, 0
    for x in range(int(dimensions[0]/length)):
        for y in range(int(dimensions[1]/length)):
            save = image[picX: picX+length, picY: picY+length, :]
            segments.append(Segment(save))
            if save_segments:
                misc.imsave(str(x)+"_"+str(y)+".png", save)
            picY+=length
        picX+=length
        picY=0
    return segments   

def calculateScores(segment_list):
    for segment1 in segment_list:
        for segment2 in segment_list:
            segment1.calculateScore(segment2)

def main():
    parser = setUpArguments()
    segment_list=breakUpImage(parser.pic, parser.number,parser.save)
    calculateScores(segment_list)
    print("I STILL WORK")


if __name__ == '__main__':
    main()