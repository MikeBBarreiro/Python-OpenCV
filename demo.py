import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

#print(sys.executable)

def hsvColorSegmentation():
    root = os.getcwd() #get root file
    imgPath = os.path.join(root, 'img/balloons.jpg')
    img = cv.imread(imgPath)
    ImgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)#main command to convert to HSV color space.

    #Look up HSC cone to understand this better if memory is foggy on the topic.
    lowerBound = np.array([172,100,50]) #lower end of the HSV values. Bright color RED
    upperBound = np.array([180,255,255]) #upper end of the HSV values. Darker color RED
    #lowerBound = np.array([106,100,92]) #lower end of the HSV values. Bright color BLUE
    #upperBound = np.array([120,255,255]) #upper end of the HSV values. Darker color BLUE
    #lowerBound = np.array([25,40,80]) #lower end of the HSV values. Bright color YELLOW
    #upperBound = np.array([31,255,255]) #upper end of the HSV values. Darker color YELLOW

    mask = cv.inRange(hsv, lowerBound, upperBound) #This will take the ranges and give us a MASK value.

    plt.figure()
    plt.imshow(ImgRGB)
    plt.show()

    cv.imshow('mask', mask)
    cv.waitKey(0) #waits 0 seconds (infinite) for user to hit a key for the app to continue.


def captureComputerVision():
    cap = cv.VideoCapture(0) #connect to caputre device, the number takes Device Number

    while cap.isOpened():
        ret, frame = cap.read() #get a frame from the capture device.

        if(ret): #if ret is false, it (it = device #) could be that the device is in use or it jsut wont work for the camera.
            cv.imshow('WebCam', frame)
            #plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) ## returngs true rgb color.

        #breaks the while.
        if cv.waitKey(1) & 0xFF == ord('z'):
            break

    cap.release()
    cv.destroyAllWindows()


def takePicture_FromWebCame(TakePic):
    if(TakePic):
        cap = cv.VideoCapture(0)
        ret, frame = cap.read()
        #plt.imshow(frame)
        #cap.show()
        cv.imwrite('webcamphoto.jpg', frame)
        cap.release()


def live_coloredObject_detection():

    lower = np.array([15,150,20]) #yellow lower limit
    upper = np.array([35,255,255])#yellow lower limit

    blue_lower = np.array([113,150,0])
    blue_upper = np.array([120,255,255])
    # blue_lower = np.array([112,150,20])
    # blue_upper = np.array([124,255,255])
    # blue_lower = np.array([110,50,50])
    # blue_upper = np.array([130,255,255])

    red_lower = np.array([136,150,20])
    red_upper = np.array([180,255,255])

    capture = cv.VideoCapture(0)

    while capture.isOpened():
        returned, img = capture.read()
        
        frame = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        #webCamImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = cv.inRange(frame, lower, upper)
        blue_mask = cv.inRange(frame, blue_lower, blue_upper)
        red_mask = cv.inRange(frame, red_lower, red_upper)

        contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) != 0):
            for contour in contours:
                if cv.contourArea(contour) > 500: #Dont detect random while small spots in our Contour detection. only find areas greater then 500 pixals
                    x, y, w, h = cv.boundingRect(contour) #This draw a rectangle around our contoured object and return cords of the rectanlge.
                    #-----------------1st args: The frame
                    #-----------------2nd args: TOPLEFT CORNOR CORDS ON WHERE THE REC WILL BE DRAWN
                    #-----------------3rd args: TUPLE OF BOTTOM RIGHT CORNOR CORDS ON OF THE RECTANGLE
                    #-----------------4rd args: The Objects color in BGR format
                    #-----------------5th args: Thickness in pixals of the rectanle
                    img = cv.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 3) #draw the rectangle on img variable
                    cv.putText(img, 'YELLOW DETECTION', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #put label above rectanle.
        

        contours, hierarchy = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) != 0):
            for contour in contours:
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    img = cv.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 3)
                    cv.putText(img, 'BLUE DETECTION', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,102,0), 2)
        

        contours, hierarchy = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) != 0):
            for contour in contours:
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    img = cv.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 3)
                    cv.putText(img, 'RED DETECTION', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (13,102,255), 2)
        

        #cv.imshow("Mask", mask)
        cv.imshow("WebCam", img)

        if cv.waitKey(1) & 0xFF == ord('z'):
            break

    capture.release()
    cv.destroyAllWindows()

if(__name__ == '__main__'):
    user_prompt = input("Press q for Camera Access, w for image masking, or e to take a picture!")
    
    if(user_prompt == 'q'):
        captureComputerVision()
    if(user_prompt == 'w'):
        hsvColorSegmentation()
    if(user_prompt == 'e'):
        takePicture_FromWebCame(True)
    if(user_prompt == 'f'):
        live_coloredObject_detection()