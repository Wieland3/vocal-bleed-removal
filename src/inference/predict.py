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
        checkpoint_path = constants.CHECKPOINTS_DIR + "/full_train/cp.ckpt"
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

    for i, (X_chunk, _) in enumerate(dataset.sample_generator(X, X)):
        X_chunk_batch = np.expand_dims(X_chunk, axis=0)
        y_pred_chunk = model.predict(X_chunk_batch)['vocals']
        y_pred_chunk = y_pred_chunk.squeeze(0)
        pred.append(y_pred_chunk)

    pred = np.concatenate(pred, axis=0)

    if exploited:
        pred = audio_utils.stereo_to_mono(pred)

    return pred


def get_ground_truth(y):
    """
    Gets the ground truth for a song.
    :param y: vocal for song
    :return: ground truth
    """
    gt = []
    y = audio_utils.zero_pad(y)

    for i, (_, y_chunk) in enumerate(dataset.sample_generator(y, y)):
        gt.append(y_chunk)

    gt = np.concatenate(gt, axis=0)

    return gt


if __name__ == "__main__":

    exploited = False

    piano, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Piano.wav")
    piano = audio_utils.stereo_to_mono(piano)
    guitar, _ = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Guitar.wav")
    guitar = np.expand_dims(guitar, axis=-1)
    vocals, sr = sf.read(constants.TRACKS_DIR + "/thomas/night/tracks/Voice.wav")

    clean_sources = np.add(piano * 0.5, guitar * 0.5)
    vocals = np.expand_dims(vocals, axis=-1)

    if not exploited:
        X = vocals
    else:
        X = np.hstack([vocals, clean_sources])

    prediction = predict_song(X, exploited)
    gt = get_ground_truth(vocals)
    prediction = prediction.squeeze(axis=-1)

    audio_utils.save_array_as_wave(clean_sources, constants.DEBUGGING_DATA_DIR + "/clean_sources.wav")
    audio_utils.save_array_as_wave(prediction, constants.DEBUGGING_DATA_DIR + "/full_train.wav")
    audio_utils.save_array_as_wave(gt, constants.DEBUGGING_DATA_DIR + "/GT.wav")






