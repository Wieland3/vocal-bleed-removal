import tensorflow as tf
import numpy as np
from src import constants
import soundfile as sf
from src.dataset import dataset
from src.audio_utils import audio_utils
import sys

sys.path.insert(0, constants.WAVE_UNET)
from wave_u_net import wave_u_net

if __name__ == "__main__":
    test = dataset.DataSet(subsets="test")

    checkpoint_path = constants.CHECKPOINTS_DIR + "/exploit_full_train/cp.ckpt"
    model = tf.keras.models.load_model(checkpoint_path)

    """
    song_index = 2
    song = test.songs[song_index]
    vocals = test.vocals[song_index]
    """
    pred = []
    gt = []

    piano, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Piano.wav")
    piano = audio_utils.stereo_to_mono(piano)
    guitar, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Guitar.wav")
    guitar = np.expand_dims(guitar, axis=-1)

    bleed = np.add(piano * 0.5, guitar * 0.5)

    vocals, sr = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Voice.wav")

    vocals = np.expand_dims(vocals, axis=-1)

    audio_utils.save_array_as_wave(bleed, constants.DEBUGGING_DATA_DIR + "/bleed.wav")

    X = np.hstack([vocals, bleed])

    for i, (X_chunk, y_chunk) in enumerate(test.song_data_generator(X, vocals)):
        X_chunk_batch = np.expand_dims(X_chunk, axis=0)
        y_pred_chunk = model.predict(X_chunk_batch)['vocals'].squeeze(0)
        pred.append(y_pred_chunk)
        gt.append(y_chunk)

    pred = audio_utils.stereo_to_mono(np.concatenate(pred, axis=0))
    gt = np.concatenate(gt, axis=0)

    audio_utils.save_array_as_wave(pred, constants.DEBUGGING_DATA_DIR + "/pred_exploit_3.wav")
    audio_utils.save_array_as_wave(gt, constants.DEBUGGING_DATA_DIR + "/GT.wav")






