"""
This file contains the code for handling the musdb dataset.
It implements the MusDataHandler class that handles saving and reloading np arrays of the stems.
"""


import musdb
import numpy as np
import os
from scipy.signal import convolve
import warnings
import soundfile as sf
from src import constants
from src.audio_utils.audio_utils import stereo_to_mono, normalize_target_loudness


class MusDataHandler:
    def __init__(self, root=constants.MUSDB_DIR, subsets='train', use_artificial=False):
        """
        Initializes the MusDataHandler Class.
        If a saved npz file exists, it uses it to load the data.
        :param root: Path to the musdb dataset
        :param subsets: "train" / "test" for musdb data, "art_train" / "art_test" for artificial data.
        """
        # Field if artificial dataset is used or not.
        self.art = use_artificial
        self.subsets = subsets

        if subsets == "train":
            if not self.art:
                self.path_to_npz = constants.PATH_TO_RAW_TRAIN_MUSDB_NPZ
            else:
                self.path_to_npz = constants.PATH_TO_ART_TRAIN_MUSDB_NPZ
        elif subsets == "test":
            if not self.art:
                self.path_to_npz = constants.PATH_TO_RAW_TEST_MUSDB_NPZ
            else:
                self.path_to_npz = constants.PATH_TO_ART_TEST_MUSDB_NPZ

        if os.path.exists(self.path_to_npz):
            self.X, self.y = self.load_object_arrays_from_npz()
        else:
            self.mus = musdb.DB(root=root, subsets=subsets)
            self.X, self.y = self.stems_to_npz()

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

            rir, sr = sf.read(constants.RIRS_DIR + "/RIR1.wav")
            rir = rir.reshape(-1, 1)
            convolved = convolve(other_mono, rir, mode='same')

            loudness_normalized_other = normalize_target_loudness(convolved, -37)
            loudness_normalized_other = np.clip(loudness_normalized_other, -1, 1)

            #loudness_normalized_vocal = normalize_target_loudness(vocals_mono, -20)
            #loudness_normalized_vocal = np.clip(loudness_normalized_vocal, -1, 1)

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
        return False
        if self.art:
            if self.subsets == "train":
                if index not in constants.TRAIN_FEMALE_VOCS:
                    return True
            elif self.subsets == "test":
                if index not in constants.VALID_FEMALE_VOCS:
                    return True
        return False

    def stems_to_npz(self):
        """
        Creates npz file of musdb dataset.
        :return: X, y as arrays
        """
        X = []
        y = []

        for i, track in enumerate(self.mus):
            if self.should_skip(i):
                continue

            if track.rate == 44100:
                mix, vocals = self.edit_mixture(track)
                X.append(mix)
                y.append(vocals)

        X_obj_array = np.empty((len(X),), dtype=object)
        y_obj_array = np.empty((len(y),), dtype=object)

        for i in range(len(X)):
            X_obj_array[i] = X[i]
            y_obj_array[i] = y[i]

        np.savez(self.path_to_npz, X=X_obj_array, y=y_obj_array)
        return X_obj_array, y_obj_array

    def load_object_arrays_from_npz(self):
        """
        Load object arrays X and y from a .npz file.
        :param path: path to npz file with arrays of songs
        :return: tuple with X, y where X is array of mix and y array of vocals
        """
        with np.load(self.path_to_npz, allow_pickle=True, mmap_mode='r') as data:
            X = data['X']
            y = data['y']
        return X, y
