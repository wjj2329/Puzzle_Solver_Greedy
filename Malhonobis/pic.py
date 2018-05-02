import numpy as np
import uuid
class Pic:
    def __init__(self, pic, name,top, left, right, bottom, top2, left2, right2, bottom2):
      self.pic = pic
      self.name=name
      self.up=top
      self.left=left
      self.right=right
      self.down=bottom
      self.up2=top2
      self.left2=left2
      self.right2=right2
      self.down2=bottom2
      self.covarianceTop=None
      self.covarianceBottom=None
      self.covarianceRight=None
      self.covarianceLeft=None
      self.gradleft_red=None
      self.gradleft_green=None
      self.gradleft_blue=None
      self.gradright_red=None
      self.gradright_green=None
      self.gradright_blue=None
      self.gradup_red=None
      self.gradup_green=None
      self.gradup_blue=None
      self.graddown_red=None
      self.graddown_green=None
      self.graddown_blue=None
      self.UUID=None
    def setcovariance(self):
       self.covarianceTop=np.zeros([3,3])
       self.covarianceBottom=np.zeros([3,3])
       self.covarianceRight=np.zeros([3,3])
       self.covarianceLeft=np.zeros([3,3])

       redav1up=np.average(self.up[:,0])#extract red blue and green
       greenav1up=np.average(self.up[:,1])
       blueav1up=np.average(self.up[:,2])
       redav2up=np.average(self.up2[:,0])
       greenav2up=np.average(self.up2[:,1])
       blueav2up=np.average(self.up2[:,2])

       r1=self.up[:,0]-redav1up
       r2=self.up2[:,0]-redav2up
       g1=self.up[:,1]-greenav1up
       g2=self.up2[:,1]-greenav2up
       b1=self.up[:,2]-blueav1up
       b2=self.up2[:,2]-blueav2up
       self.gradup_red=r1
       self.gradup_green=g1
       self.gradup_blue=b1
       size=self.up2[:,2].size
       size/=1.0       
       r1r2=np.dot(r1, r2)/size
       r1g2=np.dot(r1, g2)/size
       r1b2=np.dot(r1, b2)/size
       g1g2=np.dot(g1, g2)/size
       g1b2=np.dot(g1, b2)/size
       b1b2=np.dot(b1, b2)/size

       self.covarianceTop[0,0]=r1r2 #top row
       self.covarianceTop[0,1]=r1g2
       self.covarianceTop[0,2]=r1b2

       self.covarianceTop[1,0]=r1g2#middle row
       self.covarianceTop[1,1]=g1g2
       self.covarianceTop[1,2]=g1b2

       self.covarianceTop[2,0]=r1b2
       self.covarianceTop[2,1]=g1b2
       self.covarianceTop[2,2]=b1b2

       redav1down=np.average(self.down[:,0]) #extract red blue and green
       greenav1down=np.average(self.down[:,1])
       blueav1down=np.average(self.down[:,2])
       redav2down=np.average(self.down2[:,0])
       greenav2down=np.average(self.down2[:,1])
       blueav2down=np.average(self.down2[:,2])

       r1=self.down[:,0]-redav1down
       r2=self.down2[:,0]-redav2down
       g1=self.down[:,1]-greenav1down
       g2=self.down2[:,1]-greenav2down
       b1=self.down[:,2]-blueav1down
       b2=self.down2[:,2]-blueav2down
       self.graddown_red=r1
       self.graddown_green=g1
       self.graddown_blue=b1
       size=self.down2[:,2].size
       size/=1.0
       r1r2=np.dot(r1, r2)/size
       r1g2=np.dot(r1, g2)/size
       r1b2=np.dot(r1, b2)/size
       g1g2=np.dot(g1, g2)/size
       g1b2=np.dot(g1, b2)/size
       b1b2=np.dot(b1, b2)/size

       self.covarianceBottom[0,0]=r1r2 #Bottom row
       self.covarianceBottom[0,1]=r1g2
       self.covarianceBottom[0,2]=r1b2

       self.covarianceBottom[1,0]=r1g2#middle row
       self.covarianceBottom[1,1]=g1g2
       self.covarianceBottom[1,2]=g1b2

       self.covarianceBottom[2,0]=r1b2
       self.covarianceBottom[2,1]=g1b2
       self.covarianceBottom[2,2]=b1b2


       redav1left=np.average(self.left[:,0])
       greenav1left=np.average(self.left[:,1])
       blueav1left=np.average(self.left[:,2])
       redav2left=np.average(self.left2[:,0])
       greenav2left=np.average(self.left2[:,1])
       blueav2left=np.average(self.left2[:,2])

       r1=self.left[:,0]-redav1left
       r2=self.left2[:,0]-redav2left
       g1=self.left[:,1]-greenav1left
       g2=self.left2[:,1]-greenav2left
       b1=self.left[:,2]-blueav1left
       b2=self.left2[:,2]-blueav2left
       self.gradleft_red=r1
       self.gradleft_green=g1
       self.gradleft_blue=b1
       size=self.left2[:,2].size
       size/=1.0
       r1r2=np.dot(r1, r2)/size
       r1g2=np.dot(r1, g2)/size
       r1b2=np.dot(r1, b2)/size
       g1g2=np.dot(g1, g2)/size
       g1b2=np.dot(g1, b2)/size
       b1b2=np.dot(b1, b2)/size

       self.covarianceLeft[0,0]=r1r2 #Left row
       self.covarianceLeft[0,1]=r1g2
       self.covarianceLeft[0,2]=r1b2

       self.covarianceLeft[1,0]=r1g2#middle row
       self.covarianceLeft[1,1]=g1g2
       self.covarianceLeft[1,2]=g1b2

       self.covarianceLeft[2,0]=r1b2
       self.covarianceLeft[2,1]=g1b2
       self.covarianceLeft[2,2]=b1b2

       redav1right=np.average(self.right[:,0])
       greenav1right=np.average(self.right[:,1])
       blueav1right=np.average(self.right[:,2])
       redav2right=np.average(self.right2[:,0])
       greenav2right=np.average(self.right2[:,1])
       blueav2right=np.average(self.right2[:,2])
       r1=self.right[:,0]-redav1right
       r2=self.right2[:,0]-redav2right
       g1=self.right[:,1]-greenav1right
       g2=self.right2[:,1]-greenav2right
       b1=self.right[:,2]-blueav1right
       b2=self.right2[:,2]-blueav2right
       self.gradright_red=r1
       self.gradright_green=g1
       self.gradright_blue=b1
       size=self.right2[:,2].size
       size/=1.0
       r1r2=np.dot(r1, r2)/size
       r1g2=np.dot(r1, g2)/size
       r1b2=np.dot(r1, b2)/size
       g1g2=np.dot(g1, g2)/size
       g1b2=np.dot(g1, b2)/size
       b1b2=np.dot(b1, b2)/size

       self.covarianceRight[0,0]=r1r2 #right row
       self.covarianceRight[0,1]=r1g2
       self.covarianceRight[0,2]=r1b2

       self.covarianceRight[1,0]=r1g2#middle row
       self.covarianceRight[1,1]=g1g2
       self.covarianceRight[1,2]=g1b2

       self.covarianceRight[2,0]=r1b2
       self.covarianceRight[2,1]=g1b2
       self.covarianceRight[2,2]=b1b2

       self.UUID=uuid.uuid4()
