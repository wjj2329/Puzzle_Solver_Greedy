import argparse
import numpy as np
from scipy import misc


class Segment:
     pic_matrix=None
     def __init__(self,pic_matrix):
         self.pic_matrix=pic_matrix

def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pic", action="store", help="add picture you want to run on",required=True)
    parser.add_argument("-s", "--save", action="store", help="name of file you want saved ")
    parser.add_argument("-n", "--number", action="store", type=int, help="size of square segments wanted",required=True)
    return parser.parse_args()   
def breakUpImage(image,pieces):
   image=misc.imread(image)
   dimensions=image.shape
   if dimensions[0]!=dimensions[1]:
       print("Only square images will work for now ")
       exit()
   if dimensions[0]%pieces != 0 or dimensions[1]%pieces != 0:
       print("unable to break up image into equal squares")
       exit()
    segments=[]
    for()   




def main():
    parser = setUpArguments()
    breakUpImage(parser.pic, parser.number)
    print("I STILL WORK")

    


if __name__ == '__main__':
    main()