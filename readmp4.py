import numpy as np  
import cv2  
import time  
  
cap = cv2.VideoCapture('1.mp4')  
cap.read()
c = 1  
  
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)   
  
print fps  
i = 0;  
while(cap.isOpened()):  
    ret, frame = cap.read()  
      
    if not ret :  
        break  
  
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
    cv2.imshow('frame',frame)  
      
    if i%fps == 0 :  
        cv2.imwrite('image/'+str(c) + '.jpg',frame)   
    i = i+1  
    c = c+1  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
cap.release()  
cv2.destroyAllWindows()  
