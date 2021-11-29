import cv2
import numpy
import os

os.getcwd()
collection = "../images/Tesi Magistrale/FSOCO"
for dir in enumerate(os.listdir(collection)):
    single_dir = collection + "/" + dir[1] + "/img/"
    for i, filename in enumerate(os.listdir(single_dir)):
        img = cv2.imread(single_dir + filename)
        height, width = img.shape[:2]
        start_row, start_col = int(0), int(0)
        end_row, end_col = int(height), int(width)
        print("Img: " + filename)
        print("height: " + str(height) + " -- width: " + str(width))
        print("start row: " + str(start_row) + " end row: " + str(end_row) + " start col: " + str(start_col) + " end col: " + str(end_col))
        print("----------------------------------")
        cropped_image = img[start_row + 140:end_row - 140, start_col + 140:end_col - 140]
        img_name = "../images/Tesi Magistrale/FSOCO_modified/" + dir[1] + "/img/" + filename
        cv2.imwrite(img_name, cropped_image)