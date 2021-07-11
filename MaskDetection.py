# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import time 
from pygame import mixer

#system libraries
import os
import sys
from threading import Timer
import shutil
import time


lowConfidence = 0.75

#face detectinon function
def faceMaskDetectionAndPrediction(faceFrame, faceNet, maskNet):
    
    # grab the dimensions of the frame and then construct a blob
    (h, w) = faceFrame.shape[:2]
    blob = cv2.dnn.blobFromImage(faceFrame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations and the list of 
    # predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater
        # than the minimum confidence
        if confidence > lowConfidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering,
            # and preprocess it
            face = faceFrame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    return (locs, preds)

# SETTINGS
SOUND_PATH=os.getcwd()+"\\sounds\\alarm.wav" 


# Load Sounds
mixer.init()
sound = mixer.Sound(SOUND_PATH)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = r"deploy.prototxt"
weightsPath = r"res330_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face detector model...")
maskNet = load_model("Face_mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it 
    # to have a maximum width of 900 pixels
    faceFrame = vs.read()
    faceFrame = imutils.resize(faceFrame, width=900)

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = faceMaskDetectionAndPrediction(faceFrame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        if(label=="No Mask") and (mixer.get_busy()==False):
            sound.play()
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        if label =="Mask":
                print("ACCESS GRANTED")
            #arduino.write(b'H')

        else: 
                print("ACCESS DENIED")
            #arduino.write(b'L')
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(faceFrame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(faceFrame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("FaceMask Detection System By Cathbert Busiku -- k to quit", faceFrame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("k"):
        break
#import serial

#Setting up your arduino
#arduino = serial.Serial('/dev/ttyUSB0',9600)
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()