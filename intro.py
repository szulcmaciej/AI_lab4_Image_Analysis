import cv2 as cv
import numpy as np


img = cv.imread('images/room1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)

# img = cv.drawKeypoints(gray, kp, img)

# cv.imwrite('images/room1_sift.png', img)

# print(kp[0])
# cv.xfeatures2d_SIFT.compute()

kp, des = sift.detectAndCompute(gray, None)
print(des.shape)
print(len(kp))

