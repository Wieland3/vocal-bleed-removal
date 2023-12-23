import numpy as np


class NoiseGate:
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, audio):
        raise NotImplementedError("Subclass needs to implement this function.")