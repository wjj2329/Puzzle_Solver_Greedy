import numpy as np
from imageio import imread

filename = "test2 round"
filename2 = "test4 round"
for i in range(63):
    image1 = imread(filename+str(i)+".png")
    image2 = imread(filename2+str(i)+".png")
    print(np.array_equal(image1,image2))

