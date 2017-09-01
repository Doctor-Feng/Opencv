import cv2  
  
cap = cv2.VideoCapture('1.avi')  
  
cap.read() 
while True:  
    ret,frame = cap.read()
    cv2.imshow("Oto Video", frame)  
    cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()
