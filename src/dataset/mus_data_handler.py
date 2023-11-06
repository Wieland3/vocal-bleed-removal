import musdb
import numpy as np
import os
from src import constants


class MusDataHandler:
    def __init__(self, root=constants.MUSDB_DIR):
        self.path_to_npz = constants.PATH_TO_MUSDB_NPZ

        if os.path.exists(self.path_to_npz):
            self.X, self.y = self.load_object_arrays_from_npz()
        else:
            self.mus = musdb.DB(root=root)
            self.X, self.y = self.stems_to_npz()

    def stems_to_npz(self):
        """
        Creates npz file of musdb dataset.
        :return: X, y as arrays
        """
        X = []
        y = []

        for track in self.mus:
            if track.rate == 44100:
                X.append(track.audio)
                y.append(track.targets['vocals'].audio)

        X_obj_array = np.empty((len(X),), dtype=object)
        y_obj_array = np.empty((len(y),), dtype=object)

        for i in range(len(X)):
            X_obj_array[i] = X[i]
            y_obj_array[i] = y[i]

        np.savez(constants.PATH_TO_MUSDB_NPZ, X=X_obj_array, y=y_obj_array)

        return X_obj_array, y_obj_array

    def load_object_arrays_from_npz(self, path=constants.PATH_TO_MUSDB_NPZ):
        """
        Load object arrays X and y from a .npz file.
        :param path: path to npz file with arrays of songs
        :return: tuple with X, y where X is array of mix and y array of vocals
        """
        with np.load(path, allow_pickle=True) as data:
            X = data['X']
            y = data['y']
            X = np.asarray(X, dtype=object)
            y = np.asarray(y, dtype=object)
        return X, y
