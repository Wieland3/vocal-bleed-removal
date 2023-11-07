"""
This file implements the dataset class which should handle the creation of the dataset for training
and testing.
"""

import numpy as np
from src.dataset.mus_data_handler import MusDataHandler
from src.audio_utils.audio_utils import zero_pad, center_crop
from src import constants


class DataSet:
    def __init__(self):
        handler = MusDataHandler()
        self.songs, self.vocals = handler.X, handler.y

    def create_dataset(self):
        X = []
        y = []

        for i in range(len(self.songs)):

            if i > 2:
                break

            song = zero_pad(self.songs[i])
            vocal = zero_pad(self.vocals[i])

            l, r = 0, constants.N_SAMPLES_IN
            step = constants.N_SAMPLES_OUT

            while r <= song.shape[0]:
                X.append(np.array(song[l:r]))
                y.append(center_crop(vocal[l:r]))

                l += step
                r += step

        return np.array(X), np.array(y)
