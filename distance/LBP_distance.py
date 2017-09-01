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


