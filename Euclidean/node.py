from __future__ import division
import math
import numpy as np
from scipy.ndimage.morphology import *
import PIL
from info import Info
from scipy import misc


class Node:
    def __init__(self, pic, name, rotate, array, picarray, maxdimensions, mymap):
      self.pic = pic
      self.name=name
      self.rotate=rotate
      self.array=array
      self.picarray=picarray
      self.globalpiccombo=None
      self.globaldirection=None
      self.globalnode=None
      self.globalnewarray=None
      self.Evilnode=False
      self.bestpicarray=None
      self.bestpicarraytemp=None
      self.x=int(maxdimensions[0])
      self.y=int(maxdimensions[1])
      self.secondbestnode=None
      self.secondbestnodescore=None
      self.bestNode=None
      self.mapofnodeandedges=mymap
      self.haschanged=True
      self.degree=0
      self.combos=[]
      self.besti=0
      self.bestnumofcomp=0

    def getscorefortest_left(self, pic2):
      return self.scoreforleft(self.picarray[0,0].pic, pic2.picarray[0,0].pic, None,self.picarray[0,0], pic2.picarray[0,0] ,0)
    def getscorefortest_bottom(self, pic2):
      return self.scorefortop(self.picarray[0,0].pic, pic2.picarray[0,0].pic, None,self.picarray[0,0], pic2.picarray[0,0],0 )
    def scoreforleft(self, pic1, pic2, node2, location, location2, currentround): #as in pic2 terms
      score=0
      red=0
      green=0
      blue=0
      x=pic1.shape[0]
      i=0
      if (location.UUID,location2.UUID, "left") in self.mapofnodeandedges:
        return self.mapofnodeandedges[(location.UUID,location2.UUID, "left")]
      if currentround !=0:
        print "i mess up ", currentround,  "it doesn't contain these ", location.UUID, location2.UUID, "left"
        for v in self.mapofnodeandedges.keys():
          print v
        raise ValueError("Ineffiency Error")
      while i<len(pic1[0]):
                   tuple=abs(pic1[i,x-1]-pic2[i,0])
                   r=tuple[0].astype(int)
                   g=tuple[1].astype(int)
                   b=tuple[2].astype(int)
                   red+=r
                   green+=g
                   blue+=b
                   score+=math.sqrt((r*r)+(g*g)+(b*b))
                   i+=1
      self.mapofnodeandedges[(location.UUID,location2.UUID, "left")] =int (round(score))
      return score

    def scoreforright(self, pic1, pic2, node2, location, location2, currentround): #as in pic 2 bottom
      red=0
      green=0
      blue=0
      score=0
      bottomsize=0
      x=pic1.shape[0]
      i=0
      if (location.UUID,location2.UUID, "right") in self.mapofnodeandedges:
       return self.mapofnodeandedges[(location.UUID,location2.UUID, "right")]
      while i<len(pic1[0]):
                    tuple=abs(pic1[i,0]-pic2[i,x-1])
                    r=tuple[0].astype(int)
                    g=tuple[1].astype(int)
                    b=tuple[2].astype(int)
                    red+=r
                    green+=g
                    blue+=b
                    score+=math.sqrt((r*r)+(g*g)+(b*b))
                    i+=1
      self.mapofnodeandedges[(location.UUID,location2.UUID, "right")] =int (round(score))
      return int(round(score))

    def scorefortop(self, pic1, pic2, node2, location, location2, currentround):
      red=0
      green=0
      blue=0
      score=0
      x=pic1.shape[0]
      i=0
      if (location.UUID,location2.UUID, "top") in self.mapofnodeandedges:
        return self.mapofnodeandedges[(location.UUID,location2.UUID, "top")]
      while i<len(pic1[0]):
                    tuple=abs(pic1[x-1,i]-pic2[0,i])
                    r=tuple[0].astype(int)
                    g=tuple[1].astype(int)
                    b=tuple[2].astype(int)
                    score+=math.sqrt((r*r)+(g*g)+(b*b))
                    red+=r
                    green+=g
                    blue+=b
                    i+=1
      self.mapofnodeandedges[(location.UUID,location2.UUID, "top")] =int (round(score))
      return int(round(score))

    def scoreforbottom(self, pic1, pic2, node2,location, location2, currentround):  #as in pic 2's right
      red=0
      green=0
      blue=0
      score=0
      i=0
      x=pic1.shape[0]
      if (location.UUID,location2.UUID, "bottom") in self.mapofnodeandedges:
        return self.mapofnodeandedges[(location.UUID,location2.UUID, "bottom")]
      while i<len(pic1[0]):
                    tuple=abs(pic1[0,i]-pic2[x-1,i])
                    r=tuple[0].astype(int)
                    g=tuple[1].astype(int)
                    b=tuple[2].astype(int)
                    red+=r
                    green+=g
                    blue+=b
                    score+=math.sqrt((r*r)+(g*g)+(b*b))
                    i+=1
      self.mapofnodeandedges[(location.UUID,location2.UUID, "bottom")] =int (round(score))
      return int(round(score))

    def getscore(self, pair1, pair2,  nodearray1, nodearray2, direction, node2, r, c, location, currentround):
       piece1=self.picarray
       piece2=node2.picarray
       h1 = piece1.shape[0]#  the x and y of the pieces
       w1 = piece1.shape[1]
       h2 = piece2.shape[0]
       w2 = piece2.shape[1]
       oldpadded1=np.zeros( (h1+2*h2,w1+2*w2), dtype="object" )
       padded1 = np.zeros( (h1+2*h2,w1+2*w2), dtype="object" )
       padded1[h2:(h2+h1),w2:(w2+w1)] = piece1
       temp=np.zeros( (h1+2*h2,w1+2*w2), dtype="object"  )
       temp[r:(h2+r),c:(w2+c)]=piece2
       distancex=0
       distancey=0
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
          return self.scoreforright(padded1[pair1[0], pair1[1]].pic, temp[pair2[0], pair2[1]].pic, node2, temp[pair2[0], pair2[1]],padded1[pair1[0], pair1[1]], currentround )
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
       else:
           raise ValueError('INVALID DIRECTION')
           return None

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
     if temp.shape[0]>self.x or temp.shape[1]>self.y:
        return False
     return True

    def compare(self, node2, currentround):
       piece1=self.array
       piece2=node2.array
       h1 = piece1.shape[0]#  the x and y of the pieces
       w1 = piece1.shape[1]
       h2 = piece2.shape[0]
       w2 = piece2.shape[1]
       padded1 = np.zeros( (h1+2*h2,w1+2*w2) )
       padded1[h2:(h2+h1),w2:(w2+w1)] = piece1
       dilation_mask = np.asarray( [[0,1,0], [1,1,1,], [0,1,0]] )
       result = binary_dilation(input=padded1,structure=dilation_mask)
       neighboring_connections = result - padded1
       smallestx1=min(np.nonzero(padded1)[1])
       smallesty1=min(np.nonzero(padded1)[0])
       biggestx1=max(np.nonzero(padded1)[1])
       biggesty1=max(np.nonzero(padded1)[0])
       bestscore=100000000000
       bestpic=None
       direction=None
       othernode=None
       newnodegraph=None
       secondbestnodescore=100000000000
       for x in range(h1 + 2*h2 - (h2-1)):
          for y in range(w1 + 2*w2 - (w2-1)):
             pad_with_piece2 = np.zeros(neighboring_connections.shape)
             pad_with_piece2[x:(x+h2),y:(y+w2)] = piece2
             connect_map = np.logical_and(neighboring_connections,pad_with_piece2)
             overlap_map = np.logical_and(padded1,pad_with_piece2)
             #print overlap_map
             has_connections = np.sum(connect_map[:]) > 0
             has_overlap = np.sum(overlap_map[:]) > 0
             score=10000000000
             newnodegraph=padded1+pad_with_piece2
             if has_connections and not has_overlap and self.checkforcompatibility(newnodegraph): #and pad_with_piece2containsthis: This is ruins it#so paddedwith 2 is the one that changes
                newnodegraph=padded1+pad_with_piece2
                store= np.nonzero(padded1)
                score=0  #in here something is messed up!!!!!
                numofcompar=0
                for i in range(0,len(store[0])):
                    temp=[store[0][i], store[1][i]] #these are the non zero pairs for padded 1 these are correct!
                    if pad_with_piece2[temp[0]][temp[1]+1]==1:
                       numofcompar+=1
                       score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"left", node2, x, y, i, currentround)) #left of the first one
                    if pad_with_piece2[temp[0]][temp[1]-1]==1:
                       numofcompar+=1
                       score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"right", node2,  x, y, i,currentround)) #right of the first one
                    if pad_with_piece2[temp[0]+1][temp[1]]==1:
                       numofcompar+=1
                       score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"up",node2, x, y,i,currentround))#down of the first one
                    if pad_with_piece2[temp[0]-1][temp[1]]==1:
                       numofcompar+=1
                       score+=(self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"down",node2, x,y, i,currentround)) #up of the first one
                if numofcompar!=0:  #something strange here possibily when divide it gets bigger?????
                  #print numofcompar
                  score=score/numofcompar
                #print "score before ", numofcompar, score
                #score=score/(len(store[0]))
                #print "score after ", numofcompar, score
                #score=score/(i+1)
                #if currentround==1:
                #print score
                if self.globaldirection!="down" and self.globaldirection!="right":
                 allvalues=Info()
                 allvalues.secondbestnodescore=bestscore
                 allvalues.picture=self.globalpiccombo
                 allvalues.direction=self.globaldirection
                 allvalues.bestNode=self
                 allvalues.connectNode=self.globalnode
                 allvalues.combo=newnodegraph
                 allvalues.picarray=self.bestpicarraytemp
                 allvalues.secondbestnode=self.bestNode
                 allvalues.bestscore=score
                 self.combos.append(allvalues)
                if score<bestscore:
                    secondbestnodescore=bestscore
                    self.secondbestnode=self.bestNode
                    bestscore=score
                    direction=self.globaldirection
                    bestpic=self.globalpiccombo
                    self.bestNode=self.globalnode
                    self.globalnewarray=newnodegraph
                    self.bestpicarray=self.bestpicarraytemp
                    self.besti=i
                    self.bestnumofcomp=numofcompar
                elif secondbestnodescore>score:
                  secondbestnodescore=score

       x=Info()
       x.bestscore=bestscore
       x.picture=bestpic
       x.direction=direction
       x.bestNode=self
       x.connectNode=self.bestNode
       x.combo=self.globalnewarray
       x.picarray=self.bestpicarray
       x.secondbestnodescore=secondbestnodescore
       x.secondbestnode=self.secondbestnode
       x.i=self.besti
       x.numofcompar=self.bestnumofcomp
       return x
