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


def set_pos(x, y):
    pywinauto.mouse.move(coords=(x , y))

class CODAPI(GameAPI):
    def __init__(self, game=None):
        super().__init__(game=game)

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
        image = image_xp[980:908+215 , 407:407+323, :]
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