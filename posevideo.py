import cv2
import numpy as np
import mediapipe as mp
import time

#initialize for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

vid = cv2.VideoCapture('/Users/sarahmw/Downloads/running.mp4')
pTime = 0

#check if video opened successfully
if (vid.isOpened() == False):
        print("error opening video stream or file")

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
frame_size = (frame_width, frame_height)
fps = 20
output = cv2.VideoWriter('/Users/sarahmw/PycharmProjects/project/runner.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                             frame_size)

#read until video is completed
while(vid.isOpened()):
    #capture frame-by-frame
    ret, frame = vid.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    #cv2.imshow("Image", frame)
    #cv2.waitKey(1)

    if ret == True:
        #display resulting frame
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #print(results.pose_landmarks)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow('Frame', frame)
        output.write(frame)


        #press q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    #break loop
    else:
        break

#release video capture
vid.release()
output.release()

#close all frames
cv2.destroyAllWindows()

