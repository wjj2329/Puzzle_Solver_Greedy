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
def breakUpImage(image,length):
   image=misc.imread(image)
   dimensions=image.shape
   if dimensions[0] != dimensions[1]:
       print("Only square images will work for now ")
       exit()
   if dimensions[0]%length != 0 or dimensions[1]%length != 0:
       print("unable to break up image into equal squares")
       exit()
   segments = []
   x, y = 0, 0
   picX,picY = 0, 0
   for x in range(int(dimensions[0]/length)):
       for y in range(int(dimensions[1]/length)):
           save = image[picX: picX+length, picY: picY+length,:]
           print(save.shape, picX, picY, length)
           misc.imsave(str(x)+"_"+str(y)+".png", save)
           picY+=length
       picX+=length
       picY=0 





def main():
    parser = setUpArguments()
    breakUpImage(parser.pic, parser.number)
    print("I STILL WORK")

    


if __name__ == '__main__':
    main()