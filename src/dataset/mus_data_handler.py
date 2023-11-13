"""
This file contains the code for handling the musdb dataset.
It implements the MusDataHandler class that handles saving and reloading np arrays of the stems.
"""


import musdb
import numpy as np
import os
import pyloudnorm as pyln
from src import constants
from src.audio_utils.audio_utils import stereo_to_mono


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
            return track.audio
        else:
            meter = pyln.Meter(constants.SAMPLE_RATE)
            other_loudness = meter.integrated_loudness(track.targets['other'].audio)
            vocal_loudness = meter.integrated_loudness(track.targets['vocals'].audio)

            loudness_normalized_other = pyln.normalize.loudness(track.targets['other'].audio, other_loudness, -30)
            loudness_normalized_vocal = pyln.normalize.loudness(track.targets['vocals'].audio, vocal_loudness, -20.0)

            mix = loudness_normalized_other + loudness_normalized_vocal
            peak_normalized_mix = pyln.normalize.peak(mix, -1.0)
            return peak_normalized_mix

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
                X.append(stereo_to_mono(self.edit_mixture(track)))
                y.append(stereo_to_mono(track.targets['vocals'].audio))

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
