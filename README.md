# ./readmp4.py
------
```python
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
```
# ./video.py
------
```python
import cv2  
  
cap = cv2.VideoCapture('1.avi')  
  
cap.read() 
while True:  
    ret,frame = cap.read()
    cv2.imshow("Oto Video", frame)  
    cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()
```
# ./distance/distance_to_camera.py
------
```python
import numpy as np
import cv2

# import faces CascadeClassifier
face_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")


def distance_to_camera(knownWidth,focalLength,perWidth):
  return (knownWidth * focalLength) / perWidth 

def face_region(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (30,30),
                                        flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
  return faces


# Initialize the known parameters
KNOWN_DISTANCE = 35.0
KNOWN_WIDTH = 10.0
IMAGE_PATHS = ["images/1.jpg"]
image = cv2.imread(IMAGE_PATHS[0])
face = face_region(image)
focalLength = (face[0][2] * KNOWN_DISTANCE) / KNOWN_WIDTH

# run camera
cap = cv2.VideoCapture(0)
cap.read()

while True:
  ret,image = cap.read()
  #get image from camera
  faces = face_region(image)
  #mark the region with rectangle
  for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    distance = distance_to_camera(KNOWN_WIDTH,focalLength,w)

    cv2.putText(image, "%.2fcm" % (distance),
           ( x+ 1/2 * w, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX,
           2.0, (0, 255, 0), 2)
  cv2.imshow("image", image)
  if cv2.waitKey(5)%0x100 == 27:
    break
cap.release()
cv2.destroyAllWindows()


```
# ./distance/face.py
------
```python
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread('images/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
print faces
for (x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
# ./distance/LBP_distance.py
------
```python
import numpy as np
import cv2

# import faces CascadeClassifier
face_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

def find_marker(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray,(5,5),0)
  edged = cv2.Canny(gray,35,125)

  #find the contours
  (cnts,hierarchy) = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  c = max(cnts, key = cv2.contourArea)

  # compute the bounding box 
  return cv2.minAreaRect(c)

def distance_to_camera(knownWidth,focalLength,perWidth):
  return (knownWidth * focalLength) / perWidth 

def face_region(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (30,30),
                                        flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
  return faces


# Initialize the known parameters
KNOWN_DISTANCE = 35.0
KNOWN_WIDTH = 10.0
IMAGE_PATHS = ["images/1.jpg"]
image = cv2.imread(IMAGE_PATHS[0])
face = face_region(image)
focalLength = (face[0][2] * KNOWN_DISTANCE) / KNOWN_WIDTH

# run camera
cap = cv2.VideoCapture(0)
cap.read()

while True:
  ret,image = cap.read()
  #get image from camera
  faces = face_region(image)
  #mark the region with rectangle
  for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    distance = distance_to_camera(KNOWN_WIDTH,focalLength,w)

    cv2.putText(image, "%.2fcm" % (distance),
           ( x+ 1/2 * w, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX,
           2.0, (0, 255, 0), 2)
  cv2.imshow("image", image)
  if cv2.waitKey(5)%0x100 == 27:
    break
cap.release()
cv2.destroyAllWindows()


```
# ./cam_download.py
------
```python
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
```
# ./cam.py
------
```python
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
```
# ./showpic.py
------
```python
import cv2
img = cv2.imread("snapshot7.png")

ret,pic = cv2.threshold(img,235,255,cv2.THRESH_BINARY)

cv2.imshow("pic",pic)
cv2.imwrite("pic.png",pic)
cv2.waitKey(0)
```
