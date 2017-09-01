import cv2
cap = cv2.VideoCapture(0)
cap.read()

while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame',gray)

	k = cv2.waitKey(10)%0x100
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()
