import cv2
import numpy as np

class face_detection:

    def __init__(self) -> None:
        pass

    def draw_rectangle(self,image):
        faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return image

    def face_detect(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        cap.set(10,100)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, frame = cap.read()
            detected = self.draw_rectangle(frame)
            cv2.imshow('Wen cam', detected)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

L = face_detection()
L.face_detect()



