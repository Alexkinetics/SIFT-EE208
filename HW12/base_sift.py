import cv2
import numpy as np

img_orign = cv2.imread('3.jpg')
img=cv2.imread('target.jpg')
rows,cols = img.shape[:2]
gray_orign= cv2.cvtColor(img_orign,cv2.COLOR_BGR2GRAY)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 这个库需要将opencv版本倒回
sift=cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray_orign, None)
kp2, des2 = sift.detectAndCompute(gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75* n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img_orign, kp1, img, kp2, good[:20], None, flags=2)

cv2.imshow('img',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()





