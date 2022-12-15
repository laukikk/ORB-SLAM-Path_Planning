import cv2
import numpy as np
from functions import *
    

cap = cv2.VideoCapture("assets/footpath-clip-2.mp4")

  
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    image, masked = yolo(frame)
    cv2.imshow('image', image)

    imageLine, imagePerspective = changePerspective(frame)
    cv2.imshow('imageLine', imageLine)
    cv2.imshow('imagePerspective', imagePerspective)

    maskedLine, maskedPerspective = changePerspective(masked)
    cv2.imshow('maskedLine', maskedLine)
    cv2.imshow('maskedPerspective', maskedPerspective)

    bitwise = cv2.bitwise_and(frame, maskedLine, mask = None)
    cv2.imshow('bitwise', bitwise)

    markedImage, markedContours = getContours(maskedPerspective)
    cv2.imshow('markedImage', centerLines(markedImage))

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()