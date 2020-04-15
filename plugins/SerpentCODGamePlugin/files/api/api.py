from serpent.game_api import GameAPI

from serpent.input_controller import KeyboardKey, KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseButton, MouseEvent, MouseEvents

from serpent.frame_grabber import FrameGrabber

import pytesseract

import serpent.cv

import numpy as np

import skimage.io
import skimage.util
import skimage.morphology
import skimage.segmentation
import skimage.measure

import math
import time
import random

from mss import mss
import cv2

import pywinauto
import pyautogui

import os

from PIL import Image

YOLO_DIRECTORY = "models"
CONFIDENCE = 0.36
THRESHOLD = 0.22
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def set_pos(x, y):
    pywinauto.mouse.move(coords=(x , y))

class CODAPI(GameAPI):
    def __init__(self, game=None):
        super().__init__(game=game)
        self.W, self.H = None, None
        self.game_inputs = {
            "MOVEMENT": {
                "WALK LEFT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
                ],
                "SPRINT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
                ],
                "WALK RIGHT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
                ],
                "BACK": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)
                ],
                "JUMP": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_SPACE)
                ],
                "STOPPED": []
            },
            "CURSOR": {
                "MOVE1": [
                ],
                "MOVE2": [
                ] ,
                "MOVE3": [
                ],
                "MOVE4": [
                ],
                "IDLE_MOUSE": []
            },
            "FIRE": {
                "CLICK DOWN LEFT": [
                ],
                "CLICK DOWN RIGHT": [
                ],
                "CLICK UP LEFT": [
                ] ,
                "CLICK UP RIGHT": [
                ] ,
                "IDLE_FIRE": []
            }
        }
    def num_there(self, s):
        return any(i.isdigit() for i in s)
    def get_xp(self, image_xp):
        image_xp = np.array(image_xp)
        image=image_xp[ 407:407 + 215 , 980:980 + 323 , : ]
        image=cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        yellow_min=np.array([255,194,21] , np.uint8)
        yellow_max=np.array([255,194,21] , np.uint8)

        dst=cv2.inRange(image , yellow_min , yellow_max)
        no_yellow=cv2.countNonZero(dst)
        print('The number of yellow  pixels are: ' + str(no_yellow))
        return no_yellow
    def get_health(self, image):
        image = np.array(image)
        image=cv2.cvtColor(image , cv2.COLOR_BGRA2RGB)
        red_min=np.array([ 117, 54, 34] , np.uint8)
        red_max=np.array([ 117, 54, 34 ] , np.uint8)

        dst=cv2.inRange(image , red_min , red_max)
        no_red =cv2.countNonZero(dst)
        print('The number of red pixels are: ' + str(no_red))
        return no_red
    def is_dead(self , game_frame, ref):
        ssim_score=skimage.metrics.structural_similarity(
            game_frame ,
            ref ,
            multichannel=True
        )

        return ssim_score > 0.6
    def human(self, image):
        frame=np.array(image)
        frame=frame[ 440:440 + 200 , 860:860 + 200 , : ]
        frame=cv2.cvtColor(frame , cv2.COLOR_RGBA2BGR)

        if self.W is None or self.H is None:
            (H , W)=frame.shape[ : 2 ]

        frame=cv2.UMat(frame)
        blob=cv2.dnn.blobFromImage(frame , 1 / 260 , (150 , 150) ,
                                   swapRB=False , crop=False)
        net.setInput(blob)
        layerOutputs=net.forward(ln)
        boxes=[ ]
        confidences=[ ]
        classIDs=[ ]
        for output in layerOutputs:
            for detection in output:
                scores=detection[ 5: ]
                classID=0
                confidence=scores[ classID ]
                if confidence > CONFIDENCE:
                    box=detection[ 0: 4 ] * np.array([ W , H , W , H ])
                    (centerX , centerY , width , height)=box.astype("int")
                    x=int(centerX - (width / 2))
                    y=int(centerY - (height / 2))
                    boxes.append([ x , y , int(width) , int(height) ])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs=cv2.dnn.NMSBoxes(boxes , confidences , CONFIDENCE , THRESHOLD)
        if len(idxs) > 0:
            bestMatch=confidences[ np.argmax(confidences) ]

            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x , y)=(boxes[ i ][ 0 ] , boxes[ i ][ 1 ])
                (w , h)=(boxes[ i ][ 2 ] , boxes[ i ][ 3 ])

                # draw target dot on the frame
                cv2.circle(frame , (int(x + w / 2) , int(y + h / 5)) , 5 , (0 , 0 , 255) , -1)

                # draw a bounding box rectangle and label on the frame
                # color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame , (x , y) ,
                              (x + w , y + h) , (0 , 0 , 255) , 2)

                text="TARGET {}%".format(int(confidences[ i ] * 100))
                cv2.putText(frame , text , (x , y - 5) ,
                            cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 255 , 0) , 2)

                if bestMatch == confidences[ i ]:
                    return True
                else:
                    return False
        else:
            return False