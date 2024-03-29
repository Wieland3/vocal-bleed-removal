"""
This file implements the dataset class which should handle the creation of the dataset for training
and testing.
"""

import numpy as np
import tensorflow as tf
from src.dataset.musdb_data_handler import MusdbDataHandler
from src.audio_utils.audio_utils import zero_pad, center_crop
from src import constants


def sample_generator(mix, vocals):
    l, r = 0, constants.N_SAMPLES_IN
    step = constants.N_SAMPLES_OUT

    while r <= mix.shape[0]:
        X_chunk = np.array(mix[l:r])
        y_chunk = center_crop(vocals[l:r])
        yield X_chunk, y_chunk

        l += step
        r += step


class DataSet:
    def __init__(self, subsets="train", use_artificial=False, exploited=False):
        self.exploited=exploited
        handler = MusdbDataHandler(subsets=subsets, use_artificial=use_artificial, exploited=self.exploited)
        self.data = handler.data

    def data_generator(self):
        for mix, vocals in self.data:
            song = zero_pad(mix.astype(np.float16))
            vocal = zero_pad(vocals.astype(np.float16))

            yield from sample_generator(song, vocal)

    def get_tf_dataset(self):

        n_channels = 2 if self.exploited else 1
        output_signature = (
            tf.TensorSpec(shape=(constants.N_SAMPLES_IN, n_channels), dtype=tf.float16),
            tf.TensorSpec(shape=(constants.N_SAMPLES_OUT, n_channels), dtype=tf.float16)
        )

        return tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=output_signature
        ).repeat()
