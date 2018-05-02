from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import argparse
from matplotlib.ticker import FuncFormatter

from scipy import misc
import image_slicer
import random
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
from PIL import ImageTk, Image
import random
from functools import partial
import matplotlib.pyplot as plt
from fractions import gcd
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pic", action="store", help="add picture you want to run on")
parser.add_argument("-s", "--save", action="store", help="name of file you want saved ")
parser.add_argument("-n", "--number", action="store", type=int, help="number of segments")
parser.add_argument("-r", "--rotate", action="store_true",help="if you want the pictures to be rotated" )
args=parser.parse_args()
numberofsegments=args.number
picture= misc.imread(args.pic)  #command to grap picture we want save in array
print picture.shape
temp=picture[1:500, 1:10]   #this is dimension x and y of how much we want
mylist=image_slicer.slice(args.pic, numberofsegments)#this creates all pictures that I need
print len(mylist)
if len(mylist)!=numberofsegments:
   print "illegal operation"
   exit()
misc.imsave(args.save, temp)  #command to save picture to output
nodes=[]
segments=[]
globalpicturedimensions=()
maxdimensions=[]
mymap={}
temp=mylist[-1].basename
maxdimensions.append(temp.split('_')[1])
maxdimensions.append(temp.split('_')[2])
window=Tkinter.Tk()
window.title("Picture")
img=ImageTk.PhotoImage(Image.open("piginforest.jpg"))
w = Tkinter.Label(window, image = img)
for i in mylist:
   segments.append(misc.imread(i.basename+".png"))
#random.shuffle(segments)
i=0
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
      print "try this ", gcd(pic.shape[0], pic.shape[1])
      exit()
    name=args.save[:-4]+str(i)+args.save[-4:]
    misc.imsave(name, pic)
    i+=1
    globalpicturedimensions=pic.shape
    temp=Node(pic, name, args.rotate, np.asarray([[1,0],[0,0]]), np.asarray([[Pic(pic, name),0], [0,0]]), maxdimensions, mymap)
    nodes.append(temp)
i=0
print segments[0].shape
leftpairs=[]
for x in range(len(nodes)):
   if 0!=(x+1)%int((math.sqrt(len(nodes)))) or x==0:
      leftpairs.append((x, x+1))
uppairs=[]
for x in range(len(nodes)-int(math.sqrt(len(nodes)))):
   uppairs.append((x, x+int((math.sqrt(len(nodes))))))
#print leftpairs
#print uppairs
correctscores=[]

for pair in leftpairs:
   node1=nodes[pair[0]]
   node2=nodes[pair[1]]
   temp=node1.compare(node2, i)
   temp2=node2.compare(node1, i)
   temp_score=temp.bestscore/temp.secondbestnodescore+temp2.bestscore/temp2.secondbestnodescore
   correctscores.append(round(temp_score, 4))
for pair in uppairs:
   node1=nodes[pair[0]]
   node2=nodes[pair[1]]
   temp=node1.compare(node2, i)
   temp2=node2.compare(node1, i)
   temp_score=temp.bestscore/temp.secondbestnodescore+temp2.bestscore/temp2.secondbestnodescore
   correctscores.append(round(temp_score, 4))
listofscores=[]
for node1 in nodes:
 for node2 in nodes:
    if node1!=node2:
         temp=node1.compare(node2, i)
         temp2=node2.compare(node1, i)
         temp_score=temp.bestscore/temp.secondbestnodescore+temp2.bestscore/temp2.secondbestnodescore
         listofscores.append(round(temp_score, 4))
incorrectscores=[]
for score in listofscores:
   if round(score,4) not in correctscores:
      incorrectscores.append(score)
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
print "my correct scores ", correctscores
print incorrectscores
axes = plt.gca()
axes.set_ylim([0,max(incorrectscores)+1])
axes.set_xlim([-1,1])
colors={0:"red",1:"blue"}
myscore=[0.25]*len(incorrectscores)
myscores2=[0.75]*len(correctscores)
#myscore+=myscores2
#incorrectscores+=correctscores
plt.plot(myscore,incorrectscores, 'ko',markersize=2, color="blue")
plt.plot(myscores2, correctscores, 'ko',markersize=2, color="red")
plt.show()

N_points = len(incorrectscores)+len(correctscores)
n_bins = 50

# Generate a normal distribution, center at x=0 and y=5
x = incorrectscores
y = correctscores

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=False)
hist, bins = np.histogram(x, bins=n_bins)
hist2, bins2=np.histogram(y, bins=n_bins)
# We can set the number of bins with the `bins` kwarg
print bins[:-1]
axs[0].bar(bins[:-1], hist.astype(np.float32) / hist.sum())
axs[1].bar(bins2[:-1], hist2.astype(np.float32) / hist2.sum())
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

leftpairs=[]
for x in range(len(nodes)):
   if 0!=(x+1)%int((math.sqrt(len(nodes)))) or x==0:
      leftpairs.append((x, x+1))
uppairs=[]
for x in range(len(nodes)-int(math.sqrt(len(nodes)))):
   uppairs.append((x, x+int((math.sqrt(len(nodes))))))
#print leftpairs
#print uppairs
correctscores=[]

for pair in leftpairs:
   correctscores.append(int (round(nodes[pair[0]].getscorefortest_left(nodes[pair[1]]))))
for pair in uppairs:
   correctscores.append(int (round(nodes[pair[0]].getscorefortest_bottom(nodes[pair[1]]))))
print correctscores
for node1 in nodes:
 for node2 in nodes:
    if node1!=node2:
      node1.compare(node2, 0)
incorrectscores=[]
for score in mymap.values():
   if int(round(score)) not in correctscores:
      incorrectscores.append(score)

print incorrectscores
axes = plt.gca()
axes.set_ylim([0,max(incorrectscores)+1])
axes.set_xlim([-1,1])
colors={0:"red",1:"blue"}
myscore=[0.25]*len(incorrectscores)
print myscore, "before"
myscores2=[0.75]*len(correctscores)
print "this is the ones ", myscores2
#myscore+=myscores2
print myscore, "after"
#incorrectscores+=correctscores
print myscore
print incorrectscores
plt.plot(myscore,incorrectscores, 'ko',markersize=2, color="blue")
plt.plot(myscores2, correctscores, 'ko',markersize=2, color="red")
plt.show()


exit()

#exit()
#mymap.clear()
#sleep(500000)
#for k in mymap.keys():
   #print k, mymap[k]
#for node in nodes:
   #print node
#print "my correct scores are ",correctscores
#print "my incorrect scores " ,incorrectscores
while i<numberofsegments-1:
 bestscore=1000000000
 bestinfo=None
 print "the score I get for round "+str(i)
 for node1 in nodes:
  for node2 in nodes:
     if node1!=node2:
      temp=node1.compare(node2, i)
      temp2=node2.compare(node1, i)
      temp_score=temp.bestscore/temp.secondbestnodescore+temp2.bestscore/temp2.secondbestnodescore
      if temp_score<bestscore:
        bestscore=temp_score
        bestinfo=temp


 print "my map size ", len(mymap)
 whattokeep=np.nonzero(bestinfo.combo)
 #strip out zero nonsence
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
 print "i compare, ",bestinfo.numofcompar
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
#w.mainloop()
