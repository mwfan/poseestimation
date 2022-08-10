import cv2
import numpy as np

import mediapipe as mp
#initialize for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

vid = cv2.VideoCapture('/Users/sarahmw/Downloads/running.mp4')

#check if camera opened successfully
if (vid.isOpened() == False):
        print("error opening video stream or file")

#read until video is completed
while(vid.isOpened()):
    #capture frame-by-frame
    ret, frame = vid.read()
    if ret == True:
        #display resulting frame
        cv2.imshow('Frame', frame)

        #press q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    #break loop
    else:
        break

#release video capture
vid.release()

#close all frames
cv2.destroyAllWindows()