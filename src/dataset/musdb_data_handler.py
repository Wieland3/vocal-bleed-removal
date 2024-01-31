"""
This file contains the code for handling the musdb dataset.
It implements the MusDataHandler class that handles saving and reloading np arrays of the stems.
"""


import musdb
import numpy as np
from scipy.signal import convolve
import soundfile as sf
import librosa
from src import constants
from src.audio_utils.audio_utils import stereo_to_mono, normalize_target_loudness
from random import randrange, uniform, random


class MusdbDataHandler:
    def __init__(self, root=constants.MUSDB_DIR, subsets='train', use_artificial=False, exploited=False, infinite=True):
        """
        Initializes the MusDataHandler Class.
        If a saved npz file exists, it uses it to load the data.
        :param root: Path to the musdb dataset
        :param subsets: "train" / "test" for musdb data, "art_train" / "art_test" for artificial data.
        """
        # Field if artificial dataset is used or not.
        self.art = use_artificial
        self.exploited = exploited
        self.subsets = subsets
        self.mus = musdb.DB(root=root, subsets=subsets)
        self.data = self.song_data_generator(infinite=infinite)

    def get_rir(self):
        """
        Function to load and augment room impulse response
        :return: RIR as array
        """
        rir, _ = sf.read(constants.RIRS_DIR + "/RIR1.wav")
        rir = rir.reshape(-1, 1)

        if self.subsets == "train":

            if random() < 0.33:
                noise_level = uniform(0.05, 0.25)
                white_noise = np.random.normal(0, 1, rir.shape[0])
                white_noise = white_noise.reshape(-1, 1)

                rir_max = np.max(np.abs(rir))
                noise_max = np.max(np.abs(white_noise))
                scaled_noise = white_noise * (rir_max / noise_max) * noise_level

                rir = rir + scaled_noise
                rir = np.clip(rir, -1, 1)

        return rir

    def edit_mixture(self, track):
        """
        Edits the mixture to create an artificial surrogate Dataset.
        If self.art is set to false, the original mixture from the musdb dataset is returned.
        :param track: The song to edit
        :return: edited song if self.art is set to True else it returns the song unedited.
        """
        if not self.art:
            return track.audio, track.targets['vocals'].audio
        else:
            other_mono = stereo_to_mono(track.targets['other'].audio)
            vocals_mono = stereo_to_mono(track.targets['vocals'].audio)

            rir = self.get_rir()
            convolved = convolve(other_mono, rir, mode='same')

            if self.subsets == "train":
                loudness = randrange(-40, -30, 1)
            else:
                loudness = -35

            loudness_normalized_other = normalize_target_loudness(convolved, loudness)
            loudness_normalized_other = np.clip(loudness_normalized_other, -1, 1)

            mix = loudness_normalized_other + vocals_mono
            mix = np.clip(mix, -1, 1)

            if self.exploited:
                stereo_vocals = np.concatenate([vocals_mono, vocals_mono], axis=1) # convert back to mono later
                stacked_array = np.hstack([mix, other_mono])
                return stacked_array, stereo_vocals

            return mix, vocals_mono

    def should_skip(self, index):
        """
        Function to check if a specific track should be skipped.
        If artificial dataset is selected and the song is not in the whitelist it returns True.
        :param index: Index of the song.
        :return: True or False if Song should be skipped.
        """
        if self.art and self.subsets == "test":
            if index not in constants.VALID_FEMALE_VOCS:
                return True
        return False

    def song_data_generator(self, infinite=True):
        """
        Generates one song of mix, vocals.
        :yields: mix, vocals
        """
        while True:
            for i, track in enumerate(self.mus):
                if self.should_skip(i):
                    continue
                if track.rate == 44100:
                    mix, vocals = self.edit_mixture(track)
                    yield mix, vocals

            if not infinite:
                break
