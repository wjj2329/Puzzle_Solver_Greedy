import argparse
import image_slicer
import random
from scipy import misc
from PIL import ImageDraw, ImageFont
from node import Node
from nodedir import Nodedir
from info import Info
import Queue as queue
import numpy as np
from pic import Pic
import sys
from scipy.optimize import linprog
import time
import math
import Tkinter
import imageio
from PIL import ImageTk, Image

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pic", action="store", help="add picture you want to run on")
parser.add_argument("-s", "--save", action="store", help="name of file you want saved ")
parser.add_argument("-n", "--number", action="store", type=int, help="number of segments")
parser.add_argument("-r", "--rotate", action="store_true",help="if you want the pictures to be rotated" )
args=parser.parse_args()
numberofsegments=4
picture= imageio.imread("william.png")  #command to grap picture we want save in array
mylist=image_slicer.slice("william.png", numberofsegments)#this creates all pictures that I need
print len(mylist)
if len(mylist)!=numberofsegments:
   print "illegal operation"
   exit()
#misc.imsave(args.save, temp)  #command to save picture to output
nodes=[]
segments=[]
globalpicturedimensions=()
maxdimensions=[]
temp=mylist[-1].basename
maxdimensions.append(temp.split('_')[1])
maxdimensions.append(temp.split('_')[2])
window=Tkinter.Tk()
window.title("Picture")
img=ImageTk.PhotoImage(Image.open("piginforest.jpg"))
w = Tkinter.Label(window, image = img)
for i in mylist:
	segments.append(np.asarray(imageio.imread(i.basename+".png")))
#random.shuffle(segments)
#for pic in segments:
convolution=[]

myfile=open("test.txt", "w")
i=0
mymap={}
for pic in segments:
    rand=random.randint(1,4)
    degree=0
    if args.rotate==True:
       if rand==2:
		  degree=90
       if rand==3:
	      degree=180
       if rand==4:
	      degree=270
    pic=np.asarray(misc.imrotate(pic, degree, 'bicubic'), dtype=np.int16)
    if pic.shape[0]!=pic.shape[1]:
      print "invalid shape for peice size "+str(pic.shape[0])+" "+str(pic.shape[1])
      exit()
    temp="something.png"  
    name=temp[:-4]+str(i)+temp[-4:]
    imageio.imsave(name, pic)
    i+=1
    globalpicturedimensions=pic.shape
    top=pic[0,:]
    top2=pic[1,:]
    left=pic[:, 0]
    left2=pic[:,1]
    right=pic[:, pic.shape[0]-1]
    right2=pic[:, pic.shape[0]-2]
    bottom=pic[pic.shape[0]-1,:]
    bottom2=pic[pic.shape[0]-2, :]
    temp=Node(pic, name, args.rotate, np.asarray([[1,0],[0,0]]), np.asarray([[Pic(pic, name,top, left, right, bottom, top2, left2, right2, bottom2,i),0], [0,0]]), maxdimensions, mymap)
    #print "my pic is ", pic, " with right ", right, " and left ", left, " and bottom ",bottom, " and top ", top
    temp.picarray[0,0].setcovariance()
    #print temp.picarry
    nodes.append(temp)
i=0
#exit()
for node in nodes:
   for node2 in nodes:
      if node!=node2:
         node.compare(node2, 0)
#for score in mymap.values():
   #print score
while i<numberofsegments-1:
 bestscore=1000000000
 bestinfo=None
 print "the score I get for round "+str(i)
 for node1 in nodes:
  for node2 in nodes:
     if node1!=node2:
      temp=node1.compare(node2, i)
      if temp.bestscore<bestscore:
        bestscore=temp.bestscore
        bestinfo=temp




 whattokeep=np.nonzero(bestinfo.combo)
 smallestx1=min(np.nonzero(bestinfo.combo)[1])
 smallesty1=min(np.nonzero(bestinfo.combo)[0])
 biggestx1=max(np.nonzero(bestinfo.combo)[1])
 biggesty1=max(np.nonzero(bestinfo.combo)[0])
 biggest=biggestx1-smallestx1+1
 if(biggesty1-smallesty1+1>biggest):
      biggest=biggesty1-smallesty1+1
 storeing=np.zeros((biggest,biggest ), dtype="object")


 #print storing
 for y in range(0, len(whattokeep[0])):
     pair=[whattokeep[0][y], whattokeep[1][y]]
     storeing[pair[0]-smallesty1] [pair[1]-smallestx1]=bestinfo.combo[pair[0]] [ pair[1]]
 bestinfo.combo=storeing

 whattokeep=np.nonzero(bestinfo.picarray)
 #strip out zero nonsence
 smallestx1=min(np.nonzero(bestinfo.picarray)[1])
 smallesty1=min(np.nonzero(bestinfo.picarray)[0])
 biggesty1=max(np.nonzero(bestinfo.picarray)[0])
 biggest=biggestx1-smallestx1+1
 if(biggesty1-smallesty1+1>biggest):
      biggest=biggesty1-smallesty1+1
 storeing=np.zeros((biggest,biggest ),  dtype="object")
 for y in range(0, len(whattokeep[0])):
     pair=[whattokeep[0][y], whattokeep[1][y]]
     storeing[pair[0]-smallesty1] [pair[1]-smallestx1]=bestinfo.picarray[pair[0]] [ pair[1]]
 bestinfo.picarray=storeing
 zeroarray=[]
 for  x in range(globalpicturedimensions[0]):
   temp=[]
   for y in range(globalpicturedimensions[1]):
     temp.append((0,0,0))
   zeroarray.append(temp)

 listofstuff=[]
 picturetocreate=bestinfo.picarray
 for x in range(len(picturetocreate[0])):
  mylist=[]
  for y in range(len(picturetocreate)):
      temp=None
      if picturetocreate[x][y]!=0:
        temp=picturetocreate[x][y].pic
      else:
        temp= zeroarray
      mylist.append(temp)
  store=np.hstack(mylist)
  listofstuff.append(store)

 picture=np.vstack(listofstuff)

 bestinfo.bestNode.array=bestinfo.combo  #set the array of 1,0's for this node
 bestinfo.bestNode.pic=bestinfo.picture   #set the actual picture object array together
 bestinfo.bestNode.picarray=bestinfo.picarray
 nodes.remove(bestinfo.connectNode)

 print bestinfo.direction
 print bestinfo.bestNode.array
 print bestinfo.bestNode.picarray
 print bestinfo.bestscore
 print "it second best is ", bestinfo.secondbestnodescore
 myfile.write(str(bestinfo.bestscore))
 print "number of compare is ", bestinfo.numberofcompare
 print type(bestinfo.bestNode.picarray[0,0])
 misc.imsave(str(i)+".png", picture)
 myimg = str(i)+".png"
 updated_picture = ImageTk.PhotoImage(Image.open(myimg))
 w.configure(image = updated_picture)
 w.image=updated_picture
 w.pack(side = "bottom", fill = "both", expand = "no")
 window.update()
 print("i now have this number of peices total")
 print (len(nodes))
 i+=1
#sleep(7000)
w.mainloop()
