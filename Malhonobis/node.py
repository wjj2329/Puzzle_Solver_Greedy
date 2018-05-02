import math
import numpy as np
from scipy.ndimage.morphology import *
import PIL
from info import Info
from scipy import misc
from numpy.linalg import inv

class Node:
    def __init__(self, pic, name, rotate, array, picarray, maxdimensions,mymap):
      self.pic = pic
      self.name=name
      self.rotate=rotate
      self.array=array
      self.picarray=picarray
      self.globalpiccombo=None
      self.globaldirection=None
      self.globalnode=None
      self.globalnewarray=None
      self.bestpicarray=None
      self.bestpicarraytemp=None
      self.x=int(maxdimensions[0])
      self.y=int(maxdimensions[1])
      self.secondbestnode=None
      self.bestNode=None
      self.mapofnodeandedges=mymap
      self.haschanged=True
      self.degree=0
      self.combos=[]
      self.numberofcompare=0

    def scoreforleft(self, pic1, pic2, node2, location, location2, currentround): #as in pic2 terms
      score=0.0
      x=pic1.shape[0]
      i=0
      if ("left", location.UUID, location2.UUID) in self.mapofnodeandedges:
        return self.mapofnodeandedges[("left", location.UUID, location2.UUID)]
      if currentround !=0:
        raise Exception("ineffiency error")
      red=location.gradleft_red-location2.gradright_red
      green=location.gradleft_green-location2.gradright_green
      blue=location.gradleft_blue-location2.gradright_blue
      redaverage=np.average(location.gradleft_red)
      greenaverage=np.average(location.gradleft_green)
      blueaverage=np.average(location.gradleft_blue)
      cov=inv(location.covarianceLeft)
      #print redaverage, greenaverage, blueaverage

      red2=location2.gradright_red-location.gradleft_red
      green2=location2.gradright_green-location.gradleft_green
      blue2=location2.gradright_blue-location.gradleft_blue
      redaverage2=np.average(location2.gradright_red)
      greenaverage2=np.average(location2.gradright_green)
      blueaverage2=np.average(location2.gradright_blue)
      cov2=inv(location2.covarianceRight)
      while i<len(red):
           mymatrix=np.matrix([red[i]-redaverage,green[i]-greenaverage,blue[i]-blueaverage])
           mymatrix2=np.matrix([red2[i]-redaverage2,green2[i]-greenaverage2,blue2[i]-blueaverage2])
           i+=1
           score+=abs(mymatrix*cov*mymatrix.T)
           score+=abs(mymatrix2*cov2*mymatrix2.T)
      self.mapofnodeandedges[("left", location.UUID, location2.UUID)] =int (round(score))
      return score

    def scoreforright(self, pic1, pic2, node2, location, location2, currentround): #as in pic 2 right
      score=0.0
      bottomsize=0
      x=pic1.shape[0]
      i=0
      if ("right", location.UUID, location2.UUID) in self.mapofnodeandedges:
       return self.mapofnodeandedges[("right", location.UUID, location2.UUID)]
      if currentround !=0:
        print self.mapofnodeandedges
        print len(self.mapofnodeandedges)
        print "it doesn't contain these ", location.UUID, location2.UUID
        raise Exception("ineffiency error")
      red=location.gradright_red-location2.gradleft_red
      green=location.gradright_green-location2.gradleft_green
      blue=location.gradright_blue-location2.gradleft_blue
      redaverage=np.average(location.gradright_red)
      greenaverage=np.average(location.gradright_green)
      blueaverage=np.average(location.gradright_blue)
      cov=inv(location.covarianceRight)

      red2=location2.gradleft_red-location.gradright_red
      green2=location2.gradleft_green-location.gradright_green
      blue2=location2.gradleft_blue-location.gradright_blue
      redaverage2=np.average(location2.gradleft_red)
      greenaverage2=np.average(location2.gradleft_green)
      blueaverage2=np.average(location2.gradleft_blue)
      cov2=inv(location2.covarianceLeft)
      while i<len(red):
              mymatrix=np.matrix([red[i]-redaverage,green[i]-greenaverage,blue[i]-blueaverage])
              mymatrix2=np.matrix([red2[i]-redaverage2,green2[i]-greenaverage2,blue2[i]-blueaverage2])
              i+=1
              score+=abs(mymatrix*cov*mymatrix.T)
              score+=abs(mymatrix2*cov2*mymatrix2.T)
      self.mapofnodeandedges[("right", location.UUID, location2.UUID)] =int (round(score))
      return score

    def scorefortop(self, pic1, pic2, node2, location, location2, currentround):
      score=0.0
      x=pic1.shape[0]
      i=0
      if ("top",location.UUID, location2.UUID) in self.mapofnodeandedges:
        return self.mapofnodeandedges[("top", location.UUID, location2.UUID)]
      red=location.gradup_red-location2.graddown_red
      green=location.gradup_green-location2.graddown_green
      blue=location.gradup_blue-location2.graddown_blue
      redaverage=np.average(location.gradup_red)
      greenaverage=np.average(location.gradup_green)
      blueaverage=np.average(location.gradup_blue)
      #cov=node2.covarianceTop
      cov=inv(location.covarianceTop)

      red2=location2.graddown_red-location.gradup_red
      green2=location2.graddown_green-location.gradup_green
      blue2=location2.graddown_blue-location.gradup_blue
      redaverage2=np.average(location2.graddown_red)
      greenaverage2=np.average(location2.graddown_green)
      blueaverage2=np.average(location2.graddown_blue)
      cov2=inv(location2.covarianceBottom)
      while i<len(red):
           mymatrix=np.matrix([red[i]-redaverage,green[i]-greenaverage,blue[i]-blueaverage])
           mymatrix2=np.matrix([red2[i]-redaverage2,green2[i]-greenaverage2,blue2[i]-blueaverage2])
           i+=1
           score+=abs(mymatrix*cov*mymatrix.T)
           score+=abs(mymatrix2*cov2*mymatrix2.T)
      self.mapofnodeandedges[("top", location.UUID, location2.UUID)] =int (round(score))
      return score

    def scoreforbottom(self, pic1, pic2, node2,location, location2, currentround):  #as in pic 2's right
      score=0.0
      x=0
      i=0
      x=pic1.shape[0]
      if ("bottom", location.UUID, location2.UUID) in self.mapofnodeandedges:
        return self.mapofnodeandedges[("bottom", location.UUID,location2.UUID)]
      red=location.graddown_red-location2.gradup_red
      green=location.graddown_green-location2.gradup_green
      blue=location.graddown_blue-location2.gradup_blue
      redaverage=np.average(location.graddown_red)
      greenaverage=np.average(location.graddown_green)
      blueaverage=np.average(location.graddown_blue)
      #cov=node2.covarianceBottom
      cov=inv(location.covarianceBottom)
      red2=location2.gradup_red-location.graddown_red
      green2=location2.gradup_green-location.graddown_green
      blue2=location2.gradup_blue-location.graddown_blue
      redaverage2=np.average(location2.gradup_red)
      greenaverage2=np.average(location2.gradup_green)
      blueaverage2=np.average(location2.gradup_blue)
      cov2=inv(location2.covarianceTop)
      while i<len(red):
           mymatrix=np.matrix([red[i]-redaverage,green[i]-greenaverage,blue[i]-blueaverage])
           mymatrix2=np.matrix([red2[i]-redaverage2,green2[i]-greenaverage2,blue2[i]-blueaverage2])
           i+=1
           score+=abs(mymatrix*cov*mymatrix.T)
           score+=abs(mymatrix2*cov2*mymatrix2.T)
      self.mapofnodeandedges[("bottom",location.UUID, location2.UUID)] =int (round(score))
      return score

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
       #print temp, padded1
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
       secondbestnodescore=100000000000
       bestpic=None
       direction=None
       othernode=None
       newnodegraph=None
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
                score=0.0
                numofcompar=0.0
                for i in range(0,len(store[0])):
                    temp=[store[0][i], store[1][i]] #these are the non zero pairs for padded 1 these are correct!
                    if pad_with_piece2[temp[0]][temp[1]+1]==1:
                       numofcompar+=1.0
                       score+=self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"left", node2, x, y, i, currentround) #left of the first one
                       #if currentround==51:
                        #print "my score is for left ", score
                    if pad_with_piece2[temp[0]][temp[1]-1]==1:
                       numofcompar+=1.0
                       score+=self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"right", node2,  x, y, i, currentround) #right of the first one
                       #if currentround==51:
                        #print "my score is for right ", score
                    if pad_with_piece2[temp[0]+1][temp[1]]==1:
                       numofcompar+=1.0
                       score+=self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"up",node2, x, y,i, currentround)#down of the first one
                       #if currentround==51:
                        #print "my score is for up ", score
                    if pad_with_piece2[temp[0]-1][temp[1]]==1:
                       numofcompar+=1.0
                       score+=self.getscore(temp, (connect_map.nonzero()[0][0], connect_map.nonzero()[1][0]),padded1,pad_with_piece2,"down",node2, x,y, i, currentround) #up of the first one
                       #if currentround==51:
                        #print "my score is for down ", score
                #if numofcompar!=0:
                    #score/=numofcompar
                if currentround==51:
                   print score, newnodegraph
                   print "the number of compare is " ,numofcompar
                #score=score/(i+1*5)  #becomes prims
                if score<bestscore:
                    secondbestnodescore=bestscore
                    self.secondbestnode=self.bestNode
                    bestscore=score
                    direction=self.globaldirection
                    bestpic=self.globalpiccombo
                    self.bestNode=self.globalnode
                    self.globalnewarray=newnodegraph
                    self.bestpicarray=self.bestpicarraytemp
                    self.numberofcompare=numofcompar
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
       x.numberofcompare=self.numberofcompare
       return x
