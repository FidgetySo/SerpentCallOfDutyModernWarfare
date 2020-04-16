from queue import Queue
from threading import Thread

from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes


from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent

from serpent.logger import Loggers

from serpent.frame_grabber import FrameGrabber
from serpent.game_frame import GameFrame
import time

from mss import mss

import numpy as np


import pyautogui

from serpent.config import config

import cv2

import ctypes
import pynput

from pynput.keyboard import Key , Listener , KeyCode
from pynput import keyboard

SendInput=ctypes.windll.user32.SendInput
PUL=ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_=[ ("wVk" , ctypes.c_ushort) ,
               ("wScan" , ctypes.c_ushort) ,
               ("dwFlags" , ctypes.c_ulong) ,
               ("time" , ctypes.c_ulong) ,
               ("dwExtraInfo" , PUL) ]


class HardwareInput(ctypes.Structure):
    _fields_=[ ("uMsg" , ctypes.c_ulong) ,
               ("wParamL" , ctypes.c_short) ,
               ("wParamH" , ctypes.c_ushort) ]


class MouseInput(ctypes.Structure):
    _fields_=[ ("dx" , ctypes.c_long) ,
               ("dy" , ctypes.c_long) ,
               ("mouseData" , ctypes.c_ulong) ,
               ("dwFlags" , ctypes.c_ulong) ,
               ("time" , ctypes.c_ulong) ,
               ("dwExtraInfo" , PUL) ]


class Input_I(ctypes.Union):
    _fields_=[ ("ki" , KeyBdInput) ,
               ("mi" , MouseInput) ,
               ("hi" , HardwareInput) ]


class Input(ctypes.Structure):
    _fields_=[ ("type" , ctypes.c_ulong) ,
               ("ii" , Input_I) ]


def set_pos(x , y):
    x=1 + int(x * 65536. / 1920.)
    y=1 + int(y * 65536. / 1080.)
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(x , y , 0 , (0x0001 | 0x8000) , 0 ,
                                         ctypes.cast(ctypes.pointer(extra) , ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0) , ii_)
    SendInput(1 , ctypes.pointer(command) , ctypes.sizeof(command))
class SerpentCODGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause

    def setup_play(self):
        self.environment = self.game.environments["GAME"](
            game_api=self.game.api,
            input_controller=self.input_controller,
        )
        self.game_inputs_movement = [
            {
                "name": "CONTROLS" ,
                "control_type": InputControlTypes.DISCRETE ,
                "inputs": self.game.api.combine_game_inputs([ "MOVEMENT"])
            }
        ]
        self.game_inputs_combat = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["CURSOR", "FIRE"])
            }
        ]

        self.agent_movement = PPOAgent(
            "COD_MOVEMENT" ,
            game_inputs=self.game_inputs_movement ,
            callbacks=dict(
                after_observe=self.after_agent_observe ,
                before_update=self.before_agent_update ,
                after_update=self.after_agent_update
            ) ,
            input_shape=(100 , 100) ,
            ppo_kwargs=dict(
                memory_capacity=1024 ,
                discount=0.99,
                epochs=4,
                batch_size=64,
                entropy_regularization_coefficient=0.1,
                save_steps=1024,
            ) ,
            logger=Loggers.COMET_ML ,
            logger_kwargs=dict(
                api_key=config[ "comet_ml_api_key" ] ,
                project_name="serpent-ai-cod" ,
                reward_func=self.reward
            )
        )
        self.agent_combat = PPOAgent(
            "COD_COMBAT" ,
            game_inputs=self.game_inputs_combat ,
            callbacks=dict(
                after_observe=self.after_agent_observe ,
                before_update=self.before_agent_update ,
                after_update=self.after_agent_update
            ) ,
            input_shape=(100 , 100) ,
            ppo_kwargs=dict(
                memory_capacity=1024 ,
                discount=0.99 ,
                epochs=4,
                batch_size=32,
                entropy_regularization_coefficient=0.1 ,
                save_steps=1024 ,
            ) ,
            logger=Loggers.COMET_ML ,
            logger_kwargs=dict(
                api_key=config[ "comet_ml_api_key" ] ,
                project_name="serpent-ai-cod" ,
                reward_func=self.reward
            )
        )
        self.environment.new_episode(maximum_steps=1024)

        self.frame_times = []
        self.start_t=time.time()

        self.reference_killcam = np.squeeze(self.game.sprites[ "SPRITE_KILLCAM" ].image_data)[...,:3]

        self.shooting = False
        self.moving = False

        self.mouse1 = False
        self.mouse2 = False
        self.mouse3 = False
        self.mouse4 = False
        worker=Thread(target=self.start_mouse , args=())
        worker.setDaemon(True)
        worker.start()
    def handle_play(self, game_frame, game_frame_pipeline):
        print(game_frame.frame.shape)
        valid_game_state = self.environment.update_game_state(game_frame.frame)
        if not valid_game_state:
            return None
        reward_movement, reward_combat, over_boolean = self.reward(self.environment.game_state, game_frame.frame, self.moving)
        terminal = (
            over_boolean or
            self.environment.episode_over
        )
        self.agent_movement.observe(reward=reward_movement, terminal=terminal)
        self.agent_combat.observe(reward=reward_combat , terminal=terminal)
        if not terminal:
            agent_actions_movement = self.agent_movement.generate_actions(game_frame_pipeline)
            agent_actions_combat = self.agent_combat.generate_actions(game_frame_pipeline)
            str_agent_actions_movement = str(agent_actions_movement)
            str_agent_actions_combat = str(agent_actions_combat)
            if "STOPPED" in str_agent_actions_movement:
                self.moving = False
            else:
                self.moving = True
            if "MOVE1" in str_agent_actions_combat:
                self.mouse1 = True
                self.mouse2=False
                self.mouse3=False
                self.mouse4=False
            elif "MOVE2" in str_agent_actions_combat:
                self.mouse2 = True
                self.mouse1=False
                self.mouse3=False
                self.mouse4=False
            elif "MOVE3" in str_agent_actions_combat:
                self.mouse3 = True
                self.mouse1=False
                self.mouse2=False
                self.mouse4=False
            elif "MOVE4" in str_agent_actions_combat:
                self.mouse4 = True
                self.mouse1=False
                self.mouse2=False
                self.mouse3=False
            else:
                self.mouse1=False
                self.mouse2=False
                self.mouse3=False
                self.mouse4=False
            x, y = pyautogui.position()
            if "CLICK DOWN LEFT" in str_agent_actions_combat:
                pyautogui.mouseDown(button="left", x=x, y=y)
                self.shooting = True
            elif "CLICK DOWN RIGHT" in str_agent_actions_combat:
                pyautogui.mouseDown(button="right", x=x, y=y)
            elif "CLICK UP LEFT" in str_agent_actions_combat:
                pyautogui.mouseUp(button="left", x=x, y=y)
                self.shooting = False
            elif "CLICK UP RIGHT" in str_agent_actions_combat:
                pyautogui.mouseUp(button="right", x=x, y=y)
            self.environment.perform_input(agent_actions_movement)
            self.end_t=time.time()
            self.time_taken=self.end_t - self.start_t
            self.start_t=self.end_t
            self.frame_times.append(self.time_taken)
            self.frame_times=self.frame_times[ -20: ]
            self.fps=len(self.frame_times) / sum(self.frame_times)
            print(f"FPS: {self.fps}")
        else:
            x , y=pyautogui.position()
            pyautogui.mouseUp(button="right" , x=x , y=y)
            pyautogui.mouseUp(button="left" , x=x , y=y)
            self.input_controller.handle_keys([ ])
            self.environment.clear_input()

            self.agent_movement.reset()
            self.agent_combat.reset()

            time.sleep(4.5)

            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=1024)
            
    def handle_play_pause(self):
        self.input_controller.handle_keys([])
    def reward(self, game_state, image, moving):
        enemyX, enemyY, self.there_human =  self.game.api.human(image, helper=True)
        img = np.array(image)[...,:3]
        img = img[ 60:60 + 52 , 57:57 + 211, :]
        over_check=self.game.api.is_dead(img, self.reference_killcam)
        if over_check:
            reward = -1.0
            over=True
            return reward, reward, over
        else:
            over=False
            reward_movement = 0.008
            reward_combat = 0.008
            if game_state[ "xp" ] > 0:
                reward_combat += game_state[ "xp" ] * .005
            if game_state["health_levels"] > 0:
                reward_combat -= game_state["health_levels"] * 0.05
                reward_movement -= game_state[ "health_levels" ] * 0.05
            if moving:
                reward_movement += 0.00009
            else:
                reward_movement -= 0.03
            if (enemyX < 15 and enemyX > -15) or (enemyY < 15 and enemyY > -15):
                reward_combat += .005
            return reward_movement, reward_combat, over

    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        pass

    def after_agent_update(self):
        pass
    def start_mouse(self):
        while True:
            if self.mouse1:
                set_pos(965, 540)
            elif self.mouse2:
                set_pos(960, 542)
            elif self.mouse3:
                set_pos(955 , 540)
            elif self.mouse4:
                set_pos(960 , 538)
            else:
                self.mouse1=False
                self.mouse2=False
                self.mouse3=False
                self.mouse4=False
