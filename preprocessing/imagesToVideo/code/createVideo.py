import cv2
import numpy as np
import os
 
source_path = '../../../../../Desktop/trackdriveSingleImageLR/'
img_array = []
for filename in sorted(os.listdir(source_path)):
    print(filename)
    img = cv2.imread(source_path + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('../../../../../Desktop/newlap30fpsLR.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()