import cv2


# Load Image
img = cv2.imread('./data/download.jpg')
# Image Resized
img2 = cv2.resize(img,(800,600))
# Gray Image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# EYE DETECTION MODEL
eye_detector = cv2.CascadeClassifier('./Newfolder/Cascades/haarcascade_eye.xml')
eye_detections = eye_detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=10, maxSize=(70,70))
for (x,y,w,h) in eye_detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
cv2.imshow("Image",img)
cv2.waitKey(0)