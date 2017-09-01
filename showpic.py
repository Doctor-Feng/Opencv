import cv2
img = cv2.imread("snapshot7.png")

ret,pic = cv2.threshold(img,235,255,cv2.THRESH_BINARY)

cv2.imshow("pic",pic)
cv2.imwrite("pic.png",pic)
cv2.waitKey(0)
