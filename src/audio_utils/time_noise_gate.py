from src.audio_utils.noise_gate import NoiseGate
import numpy as np


class TimeNoiseGate(NoiseGate):
    def __init__(self, threshold):
        super().__init__(threshold)

    def process(self, audio):
        return np.where((audio > self.threshold), audio, 0)

