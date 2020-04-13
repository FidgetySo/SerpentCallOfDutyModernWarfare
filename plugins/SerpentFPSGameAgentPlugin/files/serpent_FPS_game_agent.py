from serpent.game_agent import GameAgent
import time

class SerpentFPSGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        self.frame_times=[ ]
        self.start_t=time.time()

    def handle_play(self, game_frame, game_frame_pipeline):
        self.end_t=time.time()
        self.time_taken=self.end_t - self.start_t
        self.start_t=self.end_t
        self.frame_times.append(self.time_taken)
        self.frame_times=self.frame_times[ -20: ]
        self.fps=len(self.frame_times) / sum(self.frame_times)
        print(self.fps)
