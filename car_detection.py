import cv2 



car_detector = cv2.CascadeClassifier('./Newfolder/Cascades/cars.xml')
image = cv2.imread("./Newfolder/Images/car.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = car_detector.detectMultiScale(image_gray, scaleFactor= 1.03, minNeighbors=5)
for (x,y,w,h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow("Image",image)
cv2.waitKey(0)