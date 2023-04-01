import cv2
import numpy as np

def img2sketch(img):
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img=cv2.GaussianBlur(grey_img, (9,9),0)
    sketch_img=cv2.divide(grey_img,blur_img, scale=256.0)
    cv2.imshow('sketch image',sketch_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("resources/loki_1.jpg")
img2sketch(img)
