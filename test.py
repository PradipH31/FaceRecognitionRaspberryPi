#!/usr/bin/python

import cv2
import pygame
import numpy as np
import os
import RPi.GPIO as GPIO
import time
relay = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay, GPIO.OUT)
GPIO.output(relay ,0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Brennon','Pao'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
print(cam)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while (cam.isOpened()):
    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id)
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 50):
            if (id == 1):
                print('id')
                pygame.mixer.init()
                pygame.mixer.music.load("Pell Ct.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() == True:
                    continue
                time.sleep(6)
                cam.release()
                cam = cv2.VideoCapture(0)
                id = 1
                cam.set(3, 640) # set video widht
                cam.set(4, 480) # set video height
                
    
            else:
                print(id)
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
    #             GPIO.output(relay, 0)
                print("Opening Lock")
                print(id)
                
                GPIO.output(relay, 1)
                time.sleep(1)
                GPIO.output(relay, 0)
                confidence = 100
                time.sleep(5)
                cam.release()
                cam = cv2.VideoCapture(0)
                cam.set(3, 640) # set video widht
                cam.set(4, 480) # set video height
                print(cam)
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(relay, 0)
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



