import cv2
import mediapipe as mp
import time
import screen_brightness_control as sbc
import math

class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.myHands = mp.solutions.hands
        self.hands = self.myHands.Hands()  # self.mode,self.maxHands,self.detectionCon,self.trackCon
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLMs, self.myHands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=False):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lankMark in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lankMark.x * w), int(lankMark.y * h)
                lmList.append([id, cx, cy])
                if draw: cv2.circle(frame, (cx, cy), 10, (255, 12, 200), cv2.FILLED)
        return lmList

    def tipRaiseInfo(self, frame):
        tips = [4, 8, 12, 16, 20]
        lmList = self.findPosition(frame)
        tipInfo = []

        if len(lmList) != 0:
            for tip in tips:
                if tip < 5:
                    if lmList[tip][1] < lmList[tip - 2][1]:
                        tipInfo.append(0)
                    else:
                        tipInfo.append(1)
                else:
                    if lmList[tip][2] > lmList[tip - 2][2]:
                        tipInfo.append(0)
                    else:
                        tipInfo.append(1)

        return tipInfo


def main():
    cTime, pTime = 0, 0

    cap = cv2.VideoCapture(1)
    Detector = handDetector()
    dist=0
    while True:
        ret, frame = cap.read()
        Detector.findHands(frame,False)
        LMs = Detector.findPosition(frame)

        if len(LMs)!=0:
            cv2.line(frame,(LMs[4][1],LMs[4][2]),(LMs[8][1],LMs[8][2]),(255,255,0),3)
            cv2.circle(frame,(LMs[4][1],LMs[4][2]),8,(255,124,24),cv2.FILLED)
            cv2.circle(frame,(LMs[8][1],LMs[8][2]),8,(255,124,24),cv2.FILLED)
            cv2.circle(frame,((LMs[4][1]+LMs[8][1])//2,(LMs[4][2]+LMs[8][2])//2),8,(255,0,24),cv2.FILLED)

            dist = math.dist([LMs[4][1],LMs[4][2]],[LMs[8][1],LMs[8][2]])
            if dist in range(50,151):sbc.set_brightness(int(dist-50),display=0)
            if dist<50:cv2.circle(frame,((LMs[4][1]+LMs[8][1])//2,(LMs[4][2]+LMs[8][2])//2),8,(0,0,0),cv2.FILLED);sbc.set_brightness(0,display=0)
            if dist>150:cv2.circle(frame,((LMs[4][1]+LMs[8][1])//2,(LMs[4][2]+LMs[8][2])//2),8,(255,255,255),cv2.FILLED);sbc.set_brightness(100,display=0)


            print(sbc.get_brightness())



        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime

        frame = cv2.flip(frame, 1)
        #cv2.putText(frame, str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 12, 13), 3)
        cv2.putText(frame, str(int(dist-50)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 12, 13), 3)

        cv2.imshow('Web Cam', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()