import numpy as np
import tensorflow as tf


def frequency_domain_loss(y_true, y_pred):
    fft_y_true = tf.signal.fft(tf.cast(y_true[0], tf.complex64))
    fft_y_pred = tf.signal.fft(tf.cast(y_pred[0], tf.complex64))

    loss = tf.reduce_mean(tf.abs(tf.math.subtract(fft_y_true, fft_y_pred)) ** 2)

    return loss


def sdr(y_true, y_pred):
    numerator = np.sum(np.square(y_true))
    denominator = np.sum(np.square(np.subtract(y_true, y_pred)))
    return 10 * np.log10(numerator / denominator)

