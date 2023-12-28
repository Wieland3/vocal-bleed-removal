import os

# If used from colab set to true
COLAB = True

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT_DIR, "tracks")
DEBUGGING_DATA_DIR = os.path.join(TRACKS_DIR, "debugging")
WAVE_UNET = os.path.join(ROOT_DIR, "wave_u_net_tf2")
EDA_DATA_DIR = os.path.join(TRACKS_DIR, "eda")

if not COLAB:
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    MUSDB_DIR = os.path.join(TRACKS_DIR, "musdb18")
    RIRS_DIR = os.path.join(ROOT_DIR, "room_irs")
else:
    MODELS_DIR = os.path.join(ROOT_DIR, "../drive/MyDrive/thesis/models")
    MUSDB_DIR = os.path.join(ROOT_DIR, "../drive/MyDrive/thesis/musdb18")
    RIRS_DIR = os.path.join(ROOT_DIR, "../drive/MyDrive/thesis/room_irs")


CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Audio
SAMPLE_RATE = 44100
N_SAMPLES_IN = 147443
N_SAMPLES_OUT = 16389

# SONGS
VALID_FEMALE_VOCS = [11, 14, 18, 21, 26, 35, 38, 37, 42, 44, 47, 49]
TRAIN_FEMALE_VOCS = [4, 5, 11, 10, 7, 6, 17, 21, 20, 19, 18, 26, 37, 47, 48, 49,
                     56, 58, 59, 60, 66, 74, 85, 88]
