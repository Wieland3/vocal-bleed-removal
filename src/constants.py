import os

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT_DIR, "tracks")
MUSDB_DIR = os.path.join(TRACKS_DIR, "musdb18")
NPZ_DATA_DIR = os.path.join(ROOT_DIR, "npz_data")

# Files
PATH_TO_MUSDB_NPZ = os.path.join(NPZ_DATA_DIR, "musdb18.npz")