import cv2
import mediapipe as mp
import time
import os
import urllib
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
winName = 'ESP32-001 CAMERA'
cv2.namedWindow(winName,cv2.WINDOW_AUTOSIZE)
prevTime = 0
while(1):
    imgResponse = urllib.request.urlopen ('http://192.168.0.15/cam-lo.jpg') #abrimos el URL
    imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    img = cv2.imdecode (imgNp,-1) #decodificamos
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow(winName, image)
        if cv2.waitKey(5) & 0xFF == 27:
            break  
cv2.destroyAllWindows()

