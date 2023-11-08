"""
This file implements the dataset class which should handle the creation of the dataset for training
and testing.
"""

import numpy as np
import tensorflow as tf
from src.dataset.mus_data_handler import MusDataHandler
from src.audio_utils.audio_utils import zero_pad, center_crop
from src import constants


class DataSet:
    def __init__(self, subsets="train"):
        handler = MusDataHandler(subsets=subsets)
        self.songs, self.vocals = handler.X, handler.y

    def data_generator(self):
        for i in range(len(self.songs)):
            song = zero_pad(self.songs[i]).astype(np.float16)
            vocal = zero_pad(self.vocals[i]).astype(np.float16)

            yield from self.song_data_generator(song, vocal)

    def song_data_generator(self, song, vocal):
        l, r = 0, constants.N_SAMPLES_IN
        step = constants.N_SAMPLES_OUT

        while r <= song.shape[0]:
            X_chunk = np.array(song[l:r])
            y_chunk = center_crop(vocal[l:r])
            yield X_chunk, y_chunk

            l += step
            r += step

    def get_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(constants.N_SAMPLES_IN, 2), dtype=tf.float16),
            tf.TensorSpec(shape=(constants.N_SAMPLES_OUT, 2), dtype=tf.float16)
        )

        return tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=output_signature
        )
