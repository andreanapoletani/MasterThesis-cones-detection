import cv2
import numpy

cam = cv2.VideoCapture(2, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#cam = cv2.VideoCapture(-1)
# resolution set
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# FPS set
cam.set(cv2.CAP_PROP_FPS,30)
#print(cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT), cam.get(cv2.CAP_PROP_FPS))
print("Auto exposure " + str(cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)))

img_counter = 0

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Entire Image", frame)
    height, width = frame.shape[:2]

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        start_row, start_col = int(0), int(0)
        # Let's get the ending pixel coordinates (bottom right of cropped top)
        end_row, end_col = int(height * .5), int(width)
        cropped_left = frame[start_row:960, start_col:1280]
        cropped_right = frame[start_row:960, 1280:end_col]
        img_name_left = "../shared/opencv_frame_{}_left.png".format(img_counter)
        img_name_right = "../shared/opencv_frame_{}_right.png".format(img_counter)
        cv2.imwrite(img_name_left, cropped_left)
        cv2.imwrite(img_name_right, cropped_right)
        print("{} written!".format(img_name_left))
        print("{} written!".format(img_name_right))
        img_counter += 1
    elif k%256 == 99:
        # C pressed
        # setting auto exposure auto
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3); 
        print(cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))

cam.release()
cv2.destroyAllWindows()