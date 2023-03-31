import cv2
import numpy as np


def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),5)
    canny = cv2.Canny(blur,100,0)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(canny,kernel,iterations=2)
    imgThre = cv2.erode(imgDial,kernel,iterations=1)
    return imgThre


def getContours(img):
    biggest = np.array([])
    maxArea=0
    contours,heirarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>3000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest

def imgWarp(img,biggest):
    reordered = reorder(biggest)
    width,height = 480,640
    pts1 = np.float32(reordered)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOut = cv2.warpPerspective(img,matrix,(width,height))
    return imgOut

def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = mypoints.sum(1)
    
    myPointsNew[0] = mypoints[np.argmin(add)]
    myPointsNew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    myPointsNew[1] = mypoints[np.argmin(diff)]
    myPointsNew[2] = mypoints[np.argmax(diff)]

    return myPointsNew

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    imgThre = preprocessing(frame)
    imgContour = frame.copy()
    biggest = getContours(imgThre)
    if biggest.size ==0:
        cv2.imshow("Doc Scanner",imgContour)
    else :
        warped = imgWarp(frame,biggest)
        warped = warped[0:620,0:460]
        # kernel = np.array([[0, -1, 0],
        #            [-1, 5,-1],
        #            [0, -1, 0]])
        # warped = cv2.filter2D(src=warped, ddepth=-1, kernel=kernel)
        cv2.imshow("DOC",warped)
        cv2.imshow("Doc Scanner",imgContour)
    if cv2.waitKey(20) & 0xFF == ord('q'):break    

cap.release()
cv2.destroyAllWindows()