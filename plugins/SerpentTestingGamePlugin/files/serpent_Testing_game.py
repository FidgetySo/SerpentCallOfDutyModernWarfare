from serpent.game import Game

from .api.api import TestingAPI

from serpent.utilities import Singleton

from serpent.input_controller import InputControllers

from .environments.cod_environment import CODEnvironment

import time


class SerpentTestingGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = "File Explorer"

        kwargs[ "input_controller" ]=InputControllers.NATIVE_WIN32
        
        kwargs["executable_path"] = "EXECUTABLE_PATH"
        
        

        super().__init__(**kwargs)

        self.api_class = TestingAPI
        self.api_instance = None

        self.environments={
            "GAME": CODEnvironment
        }

        self.frame_transformation_pipeline_string="RESIZE:100x100|GRAYSCALE|FLOAT"
        self.environment_data = dict()

    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
        }

        return regions

