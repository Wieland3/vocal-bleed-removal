import numpy as np


class NoiseGateParent:
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, audio):
        raise NotImplementedError("Subclass needs to implement this function.")