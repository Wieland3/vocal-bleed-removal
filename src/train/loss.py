import tensorflow as tf


def frequency_domain_loss(y_true, y_pred):
    y_true = tf.reduce_mean(y_true, axis=-1)
    y_pred = tf.reduce_mean(y_pred, axis=-1)

    fft_y_true = tf.signal.fft(tf.cast(y_true, tf.complex64))
    fft_y_pred = tf.signal.fft(tf.cast(y_pred, tf.complex64))

    loss = tf.reduce_mean(tf.abs(tf.math.subtract(fft_y_true, fft_y_pred)) ** 2)

    return loss