import tensorflow as tf
from src import constants


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x


def encoder_layer(x, n_filters):
    x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=8, strides=4, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(filters=2 * n_filters, kernel_size=1, strides=1, padding='same')(x)
    x = GLU()(x)
    return x


def decoder_layer(x, n_filters, strides=4):
    x = tf.keras.layers.Conv1D(filters=2 * n_filters, kernel_size=3, strides=1, padding='same')(x)
    x = GLU()(x)
    x = tf.keras.layers.Conv1DTranspose(filters=n_filters, kernel_size=8, strides=strides, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def demucs():

    i = tf.keras.layers.Input(shape=(constants.N_SAMPLES_IN, 1))
    x = tf.keras.layers.ZeroPadding1D((1,0))(i)

    en1 = encoder_layer(x, 16)
    en2 = encoder_layer(en1, 32)
    en3 = encoder_layer(en2, 64)
    en4 = encoder_layer(en3, 128)
    en5 = encoder_layer(en4, 256)
    en6 = encoder_layer(en5, 512)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))(en6)
    x = tf.keras.layers.Dense(576, activation=None)(x)
    x = tf.keras.layers.Reshape((36, 16))(x)

    x = tf.keras.layers.Concatenate()([en6, x])
    x = decoder_layer(x, 512)
    x = tf.keras.layers.Concatenate()([en5, x])
    x = decoder_layer(x, 256)
    x = tf.keras.layers.Concatenate()([en4, x])
    x = decoder_layer(x, 128)
    x = tf.keras.layers.Concatenate()([en3, x])
    x = decoder_layer(x, 64)
    x = tf.keras.layers.Concatenate()([en2, x])
    x = decoder_layer(x, 32, strides=2)

    en1_slice = tf.keras.layers.Lambda(lambda x: x[:,9215:-9214])(en1)
    x = tf.keras.layers.Concatenate()([en1_slice, x])
    x = tf.keras.layers.Lambda(lambda x: x[:,1022:-1021])(x)
    start = (constants.N_SAMPLES_IN - constants.N_SAMPLES_OUT) // 2
    end = start + constants.N_SAMPLES_OUT
    i_slice = tf.keras.layers.Lambda(lambda x: x[:, start:end])(i)
    x = tf.keras.layers.Subtract()([i_slice, x])
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding='same', activation='tanh')(x)

    return tf.keras.models.Model(inputs=i, outputs=x)
