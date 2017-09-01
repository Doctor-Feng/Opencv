import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
outvideo = cv2.VideoWriter('output.avi',fourcc,1,(640,480))
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    outvideo.write(frame)
    cv2.imshow('frame',gray)
    k = cv2.waitKey(1)%0x100
    if k==ord('q'):
		break
cap.release()
outvideo.release()
cv2.destroyAllWindows()
