from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes

from serpent.frame_grabber import FrameGrabber

from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent
from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent
import pytesseract

from serpent.logger import Loggers

import serpent.cv
import signal
import time
import random

from mss import mss

from PIL import Image

import numpy as np

import ctypes
import pynput

import tensorflow as tf

import pyautogui
import pywinauto

import os

import scipy.io as sio

def set_pos(x, y):
    pywinauto.mouse.move(coords=(x, y))

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

        self.game_inputs = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT", "CURSOR", "FIRE"])
            }
        ]
        self.agent=PPOAgent(
            "COD" ,
            game_inputs=self.game_inputs ,
            callbacks=dict(
                after_observe=self.after_agent_observe ,
                before_update=self.before_agent_update ,
                after_update=self.after_agent_update
            ) ,
            input_shape=(100 , 100) ,
            ppo_kwargs=dict(
                memory_capacity=4096,
                discount=0.9,
                epochs=15,
                batch_size=64,
                entropy_regularization_coefficient=0 ,
                learning_rate=0.0000005 ,
                gae=True,
                epsilon=0.2 ,
                save_steps=8192
            ),
            logger=Loggers.COMET_ML ,
            logger_kwargs=dict(
                api_key="" ,
                project_name="serpent-ai-cod" ,
                reward_func=self.reward
            )
        )
        self.environment.new_episode(maximum_steps=1024)
        self.frame_times = []
        self.start_t=time.time()
        self.reference_killcam = np.squeeze(self.game.sprites[ "SPRITE_KILLCAM" ].image_data)
    def handle_play(self, game_frame, game_frame_pipeline):
        with mss() as sct:
            monitor_var = sct.monitors[1]
            self.monitor = sct.grab(monitor_var)
            valid_game_state = self.environment.update_game_state(self.monitor)
        if not valid_game_state:
            return None
        reward, over_boolean = self.reward(self.environment.game_state, self.monitor)

        terminal = (
            over_boolean or
            self.environment.episode_over
        )
        self.agent.observe(reward=reward, terminal=terminal)
        if not terminal:
            game_frame_buffer = FrameGrabber.get_frames([0], frame_type="PIPELINE")
            agent_actions = self.agent.generate_actions(game_frame_buffer)
            str_agent_actions = str(agent_actions)
            if "MOVE1" in str_agent_actions:
                set_pos(1050, 540)
            elif "MOVE2" in str_agent_actions:
                set_pos(960, 590)
            elif "MOVE3" in str_agent_actions:
                set_pos(870, 540)
            elif "MOVE4" in str_agent_actions:
                set_pos(960, 490)
            x, y = pyautogui.position()
            if "CLICK DOWN LEFT" in str_agent_actions:
                pyautogui.mouseDown(button="left", x=x, y=y)
            elif "CLICK DOWN RIGHT" in str_agent_actions:
                pyautogui.mouseDown(button="right", x=x, y=y)
            elif "CLICK UP LEFT" in str_agent_actions:
                pyautogui.mouseUp(button="left", x=x, y=y)
            elif "CLICK UP RIGHT" in str_agent_actions:
                pyautogui.mouseUp(button="right", x=x, y=y)
            self.environment.perform_input(agent_actions)
            self.end_t=time.time()
            self.time_taken=self.end_t - self.start_t
            self.start_t=self.end_t
            self.frame_times.append(self.time_taken)
            self.frame_times=self.frame_times[ -20: ]
            self.fps=len(self.frame_times) / sum(self.frame_times)
            print(self.fps)
        else:
            self.environment.clear_input()
            self.agent.reset()
            time.sleep(5)
            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=1024)
    def handle_play_pause(self):
        self.input_controller.handle_keys([])
    def reward(self, game_state, image ):
        img = np.array(image)
        img = img[ 60:60 + 52 , 57:57 + 211, :]
        over_check=self.game.api.is_dead(img, self.reference_killcam)
        if over_check:
            reward = -10.0
            over=True
            return reward , over
        else:
            over=False
            reward = 0.0001
            if game_state[ "xp" ] > 0:
                reward += game_state[ "xp" ] * .5
            if game_state["health_levels"] > 0:
                reward += -(game_state["health_levels"] * .235)
            return reward , over

    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        pass

    def after_agent_update(self):
        pass