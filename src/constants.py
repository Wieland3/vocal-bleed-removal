import os

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT_DIR, "tracks")
MUSDB_DIR = os.path.join(TRACKS_DIR, "musdb18")
NPZ_DATA_DIR = os.path.join(ROOT_DIR, "npz_data")
DEBUGGING_DATA_DIR = os.path.join(TRACKS_DIR, "debugging")

# Files
PATH_TO_RAW_TRAIN_MUSDB_NPZ = os.path.join(NPZ_DATA_DIR, "raw_train_musdb18.npz")
PATH_TO_RAW_TEST_MUSDB_NPZ = os.path.join(NPZ_DATA_DIR, "raw_test_musdb18.npz")

# Audio
SAMPLE_RATE = 44100