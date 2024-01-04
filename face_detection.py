import cv2


# Load Image
img = cv2.imread('./data/download.jpg')
# Resized Image 
img = cv2.resize(img,(800,600))
# Gray Image 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# FACE DETECTION MODEL
face_detector = cv2.CascadeClassifier('./Newfolder/Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(gray, scaleFactor = 1.09)
for (x,y,w,h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow('Haarcascade',img)
cv2.waitKey(0)


