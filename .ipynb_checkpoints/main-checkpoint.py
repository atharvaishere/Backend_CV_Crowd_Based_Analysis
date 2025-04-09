import cv2

cap =cv2.VideoCapture("dog.mp4")

ret, frame = cap.read()

cv2.imshow("Img", frame)
