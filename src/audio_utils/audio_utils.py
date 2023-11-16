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

    audio_mono = np.mean(audio_stereo, axis=1, keepdims=True)

    return audio_mono


def zero_pad(array):
    """
    Pads the input audio array with zeros both at the beginning and at the end in the time domain.
    Works for both mono and stereo signals.
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


def center_crop(array, num_samples=constants.N_SAMPLES_OUT):
    """
    Center crops an audio array.
    :param array: Array to operate on
    :param num_samples: length of the cropped audio
    :return: numpy array with cropped audio
    """
    start = (array.shape[0] - num_samples) // 2
    end = start + num_samples
    return array[start:end]


def get_loudness_db(signal):
    """
    Calculates the loudness of a signal in db.
    :param signal: The signal to operate on
    :return: db Value of signal
    """
    rms = np.sqrt(np.mean(signal**2))
    db = 20 * np.log10(rms)
    return db


def normalize_target_loudness(signal, target_db):
    """
    Changes a signal so that it has the target loudness
    :param signal: Signal to normalize loudndess to target_db
    :param target_db: Target loudness
    :return: Signal with loudness normalized to target_db
    """
    current_db = get_loudness_db(signal)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear

