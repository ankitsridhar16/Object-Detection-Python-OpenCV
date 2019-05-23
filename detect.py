import cv2
import random
import math
import numpy as np
from collections import deque
from  matplotlib import pyplot as plt


# capturing from the default webcam
cap = cv2.VideoCapture(0)
# deque to store all the points for traced path
center_points = deque()
# deque to store the redo command functionality
redo = deque()
# the range of colours to be detected
lowergreen = np.array([50,100,50])
uppergreen = np.array([90, 255, 255])
while (True):
	    # reading the frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)	
    if(ret):

        frame2 = cv2.GaussianBlur(frame, (5, 5), 0)
          
         # BGR to HSV
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
          
        mask = cv2.inRange(hsv, lowergreen, uppergreen)
		      
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
          
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        image, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if (len(contours)) > 0:

            largest_contour = max(contours, key=cv2.contourArea)

            moment = cv2.moments(largest_contour)

            center = (int(moment['m10']/moment['m00']), int(moment['m01']/moment['m00']))

            cv2.circle(frame, center, 10, (255, 255, 0), 1, cv2.LINE_AA)

            center_points.appendleft(center)

            redo.clear()

        for i in range(1, len(center_points)):
            b = random.randint(200, 245)
            g = random.randint(100, 200)
            
            if math.sqrt((center_points[i-1][0] - center_points[i][0])**2 + (center_points[i-1][1] - center_points[i][1])**2) < 50:
                cv2.line(frame, center_points[i-1], center_points[i], (b, g, 0), 4, cv2.LINE_AA)
        
        # showing the frames
        cv2.imshow('frame', frame)
        cv2.imshow('image', image)
        
        # getting the input from the keyboard
        k = cv2.waitKey(1) & 0xFF
        
        # press 'q' to quit
        if k == ord('q'):
            cap.release()
            break

        elif k == ord('u') and len(center_points) >= 3:
            for i in range(1, 4):
                temp1 = center_points.popleft()
                redo.appendleft(temp1)
            print('center', center_points)
            print('redo', redo)

        elif k == ord('r') and len(redo) >=3:
            for i in range(1, 4):
                temp2 = redo.popleft()
                center_points.appendleft(temp2)
            print('center', center_points)
            print('redo', redo)

cap.release()	
cv2.destroyAllWindows()