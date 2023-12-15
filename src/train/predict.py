import tensorflow as tf
import numpy as np
from src import constants
import soundfile as sf
from src.dataset import dataset
from src.audio_utils import audio_utils
import sys

sys.path.insert(0, constants.WAVE_UNET)
from wave_u_net import wave_u_net


def load_model(exploited):
    """
    Loads correct model.
    :param exploited: If exploited Model should be used
    :return: keras Model
    """
    if not exploited:
        checkpoint_path = constants.CHECKPOINTS_DIR + "/full_train_artificial_unexploited/cp.ckpt"
    else:
        checkpoint_path = constants.CHECKPOINTS_DIR + "/exploit_full_train/cp.ckpt"
    return tf.keras.models.load_model(checkpoint_path)


def predict_song(X, exploited):
    """
    Predicts an entire song and returns prediction
    :param X: Song to predict (if exploited needs to contain clean sources in right channel)
    :param exploited: If exploited model should be used
    :return: Unbleeded song
    """
    pred = []

    model = load_model(exploited=exploited)
    X = audio_utils.zero_pad(X)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=-1)

    for i, (X_chunk, _) in enumerate(dataset.song_data_generator(X, X)):
        X_chunk_batch = np.expand_dims(X_chunk, axis=0)
        y_pred_chunk = model.predict(X_chunk_batch)['vocals'].squeeze(0)
        pred.append(y_pred_chunk)

    pred = np.concatenate(pred, axis=0)

    if exploited:
        pred = audio_utils.stereo_to_mono(pred)

    return pred


def get_ground_truth(y):
    """
    Gets the ground truth for a song.
    :param y:
    :return:
    """
    gt = []
    y = audio_utils.zero_pad(y)

    for i, (_, y_chunk) in enumerate(dataset.song_data_generator(y, y)):
        gt.append(y_chunk)

    gt = np.concatenate(gt, axis=0)

    return gt


if __name__ == "__main__":

    exploited = True

    """
    song_index = 2
    song = test.songs[song_index]
    vocals = test.vocals[song_index]
    """

    piano, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Piano.wav")
    piano = audio_utils.stereo_to_mono(piano)
    guitar, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Guitar.wav")
    guitar = np.expand_dims(guitar, axis=-1)
    vocals, sr = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Voice.wav")

    clean_sources = np.add(piano * 0.5, guitar * 0.5)
    print("CLEAN SOURCES BEFORE", clean_sources.shape)
    clean_sources = audio_utils.zero_pad(clean_sources)
    print("CLEAN SOURCES AFTER", clean_sources.shape)
    vocals = np.expand_dims(vocals, axis=-1)
    print("VOCS BEFORE", vocals.shape)
    vocals = audio_utils.zero_pad(vocals)
    print("VOCS AFTER", vocals.shape)

    if not exploited:
        X = vocals
    else:
        X = np.hstack([vocals, clean_sources])

    prediction = predict_song(X, exploited)
    vocals, sr = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Voice.wav")
    print(vocals.shape)
    print(prediction.shape)
    audio_utils.save_array_as_wave(clean_sources, constants.DEBUGGING_DATA_DIR + "/clean_sources.wav")
    audio_utils.save_array_as_wave(prediction, constants.DEBUGGING_DATA_DIR + "/pred_exploited.wav")
    audio_utils.save_array_as_wave(vocals, constants.DEBUGGING_DATA_DIR + "/GT.wav")






