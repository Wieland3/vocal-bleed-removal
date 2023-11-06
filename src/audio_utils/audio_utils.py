"""
This file implements utility functions for dealing with audio.
"""

import numpy as np
import soundfile as sf
from src import constants


def save_array_as_wave(array, path, samplerate=constants.SAMPLE_RATE):
    """
    Save a NumPy array as an audio file in WAV format.
    :param array: NumPy array containing the audio signal to be saved.
    :param path: The file path where the WAV file will be saved.
    :param samplerate: The sample rate (in Hz) of the audio signal. Defaults to the SAMPLE_RATE from constants.
    :return: None
    """
    sf.write(path, array, samplerate)
