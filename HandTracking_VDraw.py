import cv2
import mediapipe as mp
import numpy as np
import time

myHands = mp.solutions.hands
hands = myHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime,pTime =0,0

cap = cv2.VideoCapture(1)
cap.set(10, 100)
if not cap.isOpened():print("Cannot open camera");exit()
listx,listy=[],[]
while True:
    ret, frame = cap.read()

    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    if results.multi_hand_landmarks :
        for handLMs in results.multi_hand_landmarks:
            for id,lankMark in enumerate(handLMs.landmark):
                h , w , c = frame.shape
                lmx , lmy =int(lankMark.x*w) , int(lankMark.y*h)
                if id==8 :
                    listx.append(lmx)
                    listy.append(lmy)

            #mpDraw.draw_landmarks(frame,handLMs,myHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime=cTime

    for i in range(len(listx)):
        cv2.circle(frame,(listx[i],listy[i]),10,(12,123,213),cv2.FILLED)
    frame = cv2.flip(frame,1)
    cv2.putText(frame,str(fps),(10,50),cv2.FONT_HERSHEY_PLAIN,3,(255,12,13),3)
    cv2.imshow('Web Cam', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):break
cap.release()
cv2.destroyAllWindows()
