from serpent.game import Game

from .api.api import CODAPI

from serpent.utilities import Singleton

from .environments.cod_environment import CODEnvironment

import time
import offshoot
from serpent.input_controller import InputControllers


class SerpentCODGame(Game , metaclass=Singleton):

    def __init__(self , **kwargs):
        kwargs[ "platform" ]="executable"

        kwargs[ "input_controller" ]=InputControllers.NATIVE_WIN32

        kwargs[ "window_name" ]="Call of Duty®: Modern Warfare®"

        kwargs[ "executable_path" ]="C:/Program Files (x86)/Call of Duty Modern Warfare/ModernWarfare.exe"

        super().__init__(**kwargs)

        self.api_class=CODAPI
        self.api_instance=None

        self.environments={
            "GAME": CODEnvironment
        }

        self.frame_transformation_pipeline_string="RESIZE:150x150|GRAYSCALE|FLOAT"

    @property
    def screen_regions(self):
        regions={
            "AMMO": (951 , 1645 , 998 , 1756),
            "CUSTOM_GAME": (37 , 61 , 95 , 474),
            "XP": (471 , 1023 , 495 , 1096),
            "HUD_KILLCAM": (60 , 57 , 112 , 268)
        }

        return regions