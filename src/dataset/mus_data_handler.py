"""
This file contains the code for handling the musdb dataset.
It implements the MusDataHandler class that handles saving and reloading np arrays of the stems.
"""


import musdb
import numpy as np
from scipy.signal import convolve
import soundfile as sf
from src import constants
from src.audio_utils.audio_utils import stereo_to_mono, normalize_target_loudness


class MusDataHandler:
    def __init__(self, root=constants.MUSDB_DIR, subsets='train', use_artificial=False, infinite=True):
        """
        Initializes the MusDataHandler Class.
        If a saved npz file exists, it uses it to load the data.
        :param root: Path to the musdb dataset
        :param subsets: "train" / "test" for musdb data, "art_train" / "art_test" for artificial data.
        """
        # Field if artificial dataset is used or not.
        self.art = use_artificial
        self.subsets = subsets
        self.mus = musdb.DB(root=root, subsets=subsets)
        self.data = self.data_generator(infinite=infinite)
        self.rir = self.get_rir()

    @staticmethod
    def get_rir():
        """
        Function to load room impulse response
        :return: RIR as array
        """
        rir, _ = sf.read(constants.RIRS_DIR + "/RIR1.wav")
        return rir.reshape(-1, 1)

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

            convolved = convolve(other_mono, self.rir, mode='same')

            loudness_normalized_other = normalize_target_loudness(convolved, -37)
            loudness_normalized_other = np.clip(loudness_normalized_other, -1, 1)

            mix = loudness_normalized_other + vocals_mono
            mix = np.clip(mix, -1, 1)

            stereo_vocals = np.concatenate([vocals_mono, vocals_mono], axis=1)

            stacked_array = np.hstack([mix, other_mono])
            return stacked_array, stereo_vocals

    def should_skip(self, index):
        """
        Function to check if a specific track should be skipped.
        If artificial dataset is selected and the song is not in the whitelist it returns True.
        :param index: Index of the song.
        :return: True or False if Song should be skipped.
        """
        if self.art:
            if self.subsets == "train":
                if index not in constants.TRAIN_FEMALE_VOCS:
                    return True
            elif self.subsets == "test":
                if index not in constants.VALID_FEMALE_VOCS:
                    return True
        return False

    def data_generator(self, infinite=True):
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
