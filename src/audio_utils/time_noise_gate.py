from pedalboard import Pedalboard, NoiseGate
from src import constants


class TimeNoiseGate():
    def __init__(self, threshold):
        super().__init__(threshold)
        self.board = Pedalboard([NoiseGate(threshold_db=threshold, release_ms=400)])

    def process(self, audio):
        return self.board(audio, sample_rate=constants.SAMPLE_RATE)

