from src.audio_utils.noise_gate import NoiseGate
import numpy as np


class TimeNoiseGate(NoiseGate):
    def __init__(self, threshold):
        super().__init__(threshold)

    def process(self, audio):
        threshold = 10 ** (self.threshold / 20.0)
        return np.where((np.abs(audio) > threshold), audio, 0)

