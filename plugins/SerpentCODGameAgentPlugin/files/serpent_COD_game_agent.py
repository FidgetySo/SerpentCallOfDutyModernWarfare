from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes

from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent

from serpent.logger import Loggers

from serpent.frame_grabber import FrameGrabber
from serpent.game_frame import GameFrame
import time

import numpy as np

from mss import mss

import pyautogui

from serpent.config import config

import cv2

import ctypes
import pynput

from pynput.keyboard import Key, Listener, KeyCode
from pynput import keyboard

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


def set_pos(x, y):
    x = 1 + int(x * 65536. / 1920.)
    y = 1 + int(y * 65536. / 1080.)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0,
                                           ctypes.cast(
                                               ctypes.pointer(extra),
                                               ctypes.c_void_p))
    command = pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


class SerpentCODGameAgent(GameAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause

    def setup_play(self):
        #Define environments for multiple input perfoms
        self.environment_movement = self.game.environments["GAME"](
            game_api=self.game.api,
            input_controller=self.input_controller,
        )
        self.environment_combat = self.game.environments["GAME"](
            game_api=self.game.api,
            input_controller=self.input_controller,
        )
        #
        self.game_inputs_movement = [{
            "name":
            "CONTROLS",
            "control_type":
            InputControlTypes.DISCRETE,
            "inputs":
            self.game.api.combine_game_inputs(["MOVEMENT"])
        }]
        self.game_inputs_combat = [{
            "name":
            "CONTROLS",
            "control_type":
            InputControlTypes.DISCRETE,
            "inputs":
            self.game.api.combine_game_inputs(["CURSOR", "FIRE"])
        }]
        #Define movement agent
        self.agent_movement = PPOAgent(
            "COD_MOVEMENT",
            game_inputs=self.game_inputs_movement,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update),
            input_shape=(100, 100),
            ppo_kwargs=dict(
                is_recurrent=False,
                memory_capacity=1024,
                discount=0.99,
                epochs=4,
                batch_size=32,
                entropy_regularization_coefficient=0.1,
                save_steps=1024,
            ),
            logger=Loggers.COMET_ML,
            logger_kwargs=dict(
                api_key=config["comet_ml_api_key"],
                project_name="serpent-ai-cod",
                reward_func=self.reward))
        #Define shooting/mouse agent
        self.agent_combat = PPOAgent(
            "COD_COMBAT",
            game_inputs=self.game_inputs_combat,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update),
            input_shape=(100, 100),
            ppo_kwargs=dict(
                is_recurrent=False,
                memory_capacity=1024,
                discount=0.99,
                epochs=4,
                batch_size=32,
                entropy_regularization_coefficient=0.1,
                save_steps=1024,
            ),
            logger=Loggers.COMET_ML,
            logger_kwargs=dict(
                api_key=config["comet_ml_api_key"],
                project_name="serpent-ai-cod",
                reward_func=self.reward))
        # Define episode
        self.environment_movement.new_episode(maximum_steps=1024)
        self.environment_combat.new_episode(maximum_steps=1024)
        #Define FPS Start Time for counter
        self.frame_times = []
        self.start_t = time.time()
        # Preload Reference sprite for detecting is dead
        self.reference_killcam = np.squeeze(
            self.game.sprites["SPRITE_KILLCAM"].image_data)[..., :3]
        # Pre-define is agent moving
        self.moving = False

        self.shooting = False
        self.moving = False

        self.mouse1 = False
        self.mouse2 = False
        self.mouse3 = False
        self.mouse4 = False
        worker = Thread(target=self.start_mouse, args=())
        worker.setDaemon(True)
        worker.start()

    def handle_play(self, game_frame, game_frame_pipeline):
        with mss() as sct:
            monitor_sct_number = sct.monitors[1]
            self.game_image = np.array(sct.grab(monitor_sct_number))
        valid_game_state = self.environment_movement.update_game_state(
            self.game_image)
        if not valid_game_state:
            return None
        # Get reward
        reward_movement, reward_combat, over_boolean = self.reward(
            self.environment_movement.game_state, self.game_image, self.moving)
        # Define terminal
        terminal = (over_boolean or self.environment_movement.episode_over
                    or self.environment_combat.episode_over)
        # Agent observe reward and check if agent is dead
        self.agent_movement.observe(reward=reward_movement, terminal=terminal)
        self.agent_combat.observe(reward=reward_combat, terminal=terminal)
        if not terminal:
            # Generate picked action by agent
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 1, 2, 3], frame_type="PIPELINE")
            agent_actions_movement = self.agent_movement.generate_actions(
                game_frame_buffer)
            agent_actions_combat = self.agent_combat.generate_actions(
                game_frame_buffer)
            str_agent_actions_movement = str(agent_actions_movement)
            str_agent_actions_combat = str(agent_actions_combat)
            # Check if agent is moving
            if "IDLE_FIRE" in str_agent_actions_combat:
                self.shooting = False
            elif "CLICK DOWN LEFT" in str_agent_actions_combat:
                self.shooting = True
            elif "CLICK UP LEFT" in str_agent_actions_combat:
                self.shooting = False
            if "STOPPED" in str_agent_actions_movement:
                self.moving = False
            else:
                self.moving = True
            self.environment_movement.perform_input(agent_actions_movement)
            self.environment_combat.perform_input(agent_actions_combat)
            # print fps
            self.end_t = time.time()
            self.time_taken = self.end_t - self.start_t
            self.start_t = self.end_t
            self.frame_times.append(self.time_taken)
            self.frame_times = self.frame_times[-20:]
            self.fps = len(self.frame_times) / sum(self.frame_times)
            print(f"FPS: {self.fps}")
        else:
            # End Episode and upload to comet.ml
            x, y = pyautogui.position()
            pyautogui.mouseUp(button="right", x=x, y=y)
            pyautogui.mouseUp(button="left", x=x, y=y)
            self.input_controller.handle_keys([])
            self.environment_movement.clear_input()
            self.environment_combat.clear_input()
            self.agent_movement.reset()
            self.agent_combat.reset()
            # New episodes
            self.environment_movement.end_episode()
            self.environment_movement.new_episode(maximum_steps=1024)

            self.environment_combat.end_episode()
            self.environment_combat.new_episode(maximum_steps=1024)

    def handle_play_pause(self):
        self.input_controller.handle_keys([])

    def reward(self, game_state, image, moving):
        enemyX, enemyY, self.there_human = self.game.api.human(
            image, helper=True)
        x, y = pyautogui.position()
        # Set pos to detected player from yolov3 object detection defined in game_api
        set_pos(x + enemyX, y)
        #Convert to 3 channel image for dectecting is dead
        img = np.array(image)[..., :3]
        img = img[60:60 + 52, 57:57 + 211, :]
        over_check = self.game.api.is_dead(img, self.reference_killcam)
        # Give negative feedback for dying
        if over_check:
            reward = -1.0
            over = True
            return reward, reward, over
        else:
            over = False
            reward_movement = 0.008
            reward_combat = 0.008
            # Get postive feedback for getting xp/points
            if game_state["xp"] > 0:
                reward_combat += game_state["xp"] * .008
                reward_movement += game_state["xp"] * .008
            # Count red pixels of health on screen
            if game_state["health_levels"] > 0:
                reward_combat -= game_state["health_levels"] * 0.5
                reward_movement -= game_state["health_levels"] * 0.05
            # Give minor postive feedback to movememnt agent
            if moving:
                reward_movement += 0.00009
            else:
                reward_movement -= 0.03
            # Give postive feedback if looking at a enemy player
            if (enemyX < 15 and enemyX > -15) or (enemyY < 15 and enemyY > -15):
                reward_combat += .005
            if self.there_human and self.shooting:
                reward_combat += .2
            return reward_movement, reward_combat, over

    def after_agent_observe(self):
        # Next observe step
        self.environment_movement.episode_step()
        self.environment_combat.episode_step()

    def before_agent_update(self):
        pass

    def after_agent_update(self):
        pass
