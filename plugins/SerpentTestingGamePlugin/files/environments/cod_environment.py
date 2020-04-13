from serpent.environment import Environment

from serpent.input_controller import KeyboardKey

from serpent.utilities import SerpentError

import time
import collections

import numpy as np

class CODEnvironment(Environment):

    def __init__(self, game_api=None, input_controller=None):
        super().__init__("COD Environment", game_api=game_api, input_controller=input_controller)

        self.reset()

    @property
    def new_episode_data(self):
        return {}

    @property
    def end_episode_data(self):
        return {}

    def new_episode(self, maximum_steps=None, reset=False):
        self.reset_game_state()

        time.sleep(1)

        super().new_episode(maximum_steps=maximum_steps, reset=reset)

    def end_episode(self):
        super().end_episode()

    def reset(self):
        self.reset_game_state()
        super().reset()

    def reset_game_state(self):
        self.game_state = {
            "ammo_levels": 10,
            "health_levels": 0,
            "past_hp": 0,
            "healed": False,
            "previous_xp": 0,
            "xp_check": False,
            "xp": 0,
            "steps_since_damage": 100,
            "steps": 0,
        }

    def update_game_state(self, image):
        self.game_state["steps"] += 1
        self.game_state["xp"] = self.game_api.get_xp(image)
        self.game_state["health_levels"] = self.game_api.get_health(image)
        if self.game_state["steps"] > 1024:
            self.new_episode(reset=True)
            return False

        return True