import cv2
import numpy
import os

img_counter = 0

os.getcwd()
collection = "../images"
for i, filename in enumerate(os.listdir(collection)):
    img = cv2.imread("../images/" + filename)
    height, width = img.shape[:2]
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .5), int(width)
    cropped_left = img[start_row:960, start_col:1280]
    cropped_right = img[start_row:960, 1280:end_col]
    img_name_left = "../images/opencv_frame_{}_left.png".format(img_counter)
    img_name_right = "../images/opencv_frame_{}_right.png".format(img_counter)
    cv2.imwrite(img_name_left, cropped_left)
    cv2.imwrite(img_name_right, cropped_right)
    img_counter += 1
    
