import numpy as np
import tensorflow as tf
import librosa


def l1_loss_db(y_true, y_pred):
    fft_y_true = librosa.stft(y_true, n_fft=2048)
    fft_y_pred = librosa.stft(y_pred, n_fft=2048)

    fft_y_true_db = librosa.amplitude_to_db(np.abs(fft_y_true))
    fft_y_pred_db = librosa.amplitude_to_db(np.abs(fft_y_pred))

    l1_loss_db = np.mean(np.abs(fft_y_pred_db - fft_y_true_db))

    return l1_loss_db


def sdr(y_true, y_pred):
    delta = 1e-7
    numerator = np.sum(np.square(y_true))
    denominator = np.sum(np.square(np.subtract(y_true, y_pred)))
    numerator += delta
    denominator += delta
    return 10 * np.log10(numerator / denominator)


def sdr_tf(y_true, y_pred):
    delta = 1e-7
    numerator = tf.math.reduce_sum(tf.math.square(y_true))
    denominator = tf.math.reduce_sum(tf.math.square(y_true - y_pred))
    numerator += delta
    denominator += delta
    return 10 * tf.experimental.numpy.log10(tf.math.divide(numerator, denominator))

