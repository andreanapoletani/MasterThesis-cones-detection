import os
import cv2

source_path = '../../../../../Desktop/trackdrive/'
destination_path = '../../../../../Desktop/trackdriveSingleImage/'
for i, filename in enumerate(os.listdir(source_path)):
    img = cv2.imread(source_path + filename)
    # cv2.imread() -> takes an image as an input
    h, w, channels = img.shape
    half = w//2
    right_part = img[:, half:] 
    cv2.imwrite(destination_path + filename, right_part)
    cv2.waitKey(0)