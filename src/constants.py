import os

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT_DIR, "tracks")
MUSDB_DIR = os.path.join(TRACKS_DIR, "musdb18")
NPZ_DATA_DIR = os.path.join(ROOT_DIR, "npz_data")
DEBUGGING_DATA_DIR = os.path.join(TRACKS_DIR, "debugging")
WAVE_UNET = os.path.join(ROOT_DIR, "wave-unet")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
EDA_DATA_DIR = os.path.join(TRACKS_DIR, "eda")

# Files
PATH_TO_RAW_TRAIN_MUSDB_NPZ = os.path.join(NPZ_DATA_DIR, "raw_train_musdb18.npz")
PATH_TO_RAW_TEST_MUSDB_NPZ = os.path.join(NPZ_DATA_DIR, "raw_test_musdb18.npz")

# Audio
SAMPLE_RATE = 44100
N_SAMPLES_IN = 147443
N_SAMPLES_OUT = 16389

# SONGS
VALID_FEMALE_VOCS = [0, 11, 14, 18, 21, 29, 26, 35, 38, 37, 42, 44, 47, 49]
TRAIN_FEMALE_VOCS = [4, 5, 11, 10, 7, 6, 17, 21, 20, 19, 18, 26, 32, 37, 47, 48, 49,
                     56, 58, 59, 60, 66, 69, 72, 74, 85, 88]