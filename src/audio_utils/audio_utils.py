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


def stereo_to_mono(audio_stereo):
    """
    Converts a stereo audio signal to mono by averaging the channels.
    :param audio_stereo: A numpy array with shape (samples, 2) representing the stereo audio.
    :return: A numpy array with shape (samples,) representing the mono audio.
    """
    if audio_stereo.ndim != 2 or audio_stereo.shape[1] != 2:
        raise ValueError("Input audio must be a stereo signal")

    audio_mono = np.mean(audio_stereo, axis=1)

    return audio_mono


def zero_pad(array):
    """
    Pads the input audio array with zeros both at the beginning and at the end in the time domain.
    Works for both mono and stereo signals.

    If the input array is mono (1D), zeros are added at both the beginning and the end.
    If the input array is stereo (2D), zeros are added at both the beginning and the end of each channel.

    :param array: Input audio array, which can be mono (1D) or stereo (2D).
    :return: Padded audio array.
    """
    pad_width = (constants.N_SAMPLES_IN, constants.N_SAMPLES_IN)

    if array.ndim == 1:
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    elif array.ndim == 2:
        padded_array = np.pad(array, (pad_width, (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Input array must be either mono or stereo.")

    return padded_array
