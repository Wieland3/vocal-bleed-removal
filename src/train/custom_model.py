import tensorflow as tf
from src import constants
import sys
sys.path.insert(0, constants.WAVE_UNET)
from wave_u_net import wave_u_net

params = {
  "num_initial_filters": 12,
  "num_layers": 12,
  "kernel_size": 15,
  "merge_filter_size": 5,
  "source_names": ["vocals"],
  "num_channels": 1,
  "output_filter_size": 1,
  "padding": "valid",
  "input_size": 147443,
  "context": True,
  "upsampling_type": "learned",         # "learned" or "linear"
  "output_activation": "tanh",        # "linear" or "tanh"
  "output_type": "direct",          # "direct" or "difference"
}


def down_block(x, n_filters):
    x = tf.keras.layers.Conv1D(n_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool1D()(x)
    return x

def up_block(x, n_filters):
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Conv1D(n_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def custom_model():
    inputs = tf.keras.layers.Input(shape=(147443,))

    wu = wave_u_net(**params)(inputs)

    stft = tf.signal.stft(inputs, 2048, 512)
    mag, phase = tf.abs(stft), tf.math.angle(stft)
    mag_padded = tf.keras.layers.ZeroPadding1D(2)(mag)

    d1 = down_block(mag_padded, 32)
    d2 = down_block(d1, 64)
    d3 = down_block(d2, 128)

    x = tf.keras.layers.Conv1D(128, 3, padding='same')(d3)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    u1 = up_block(x, 128)
    u1 = tf.keras.layers.Concatenate()([u1, d2])
    u2 = up_block(u1, 64)
    u2 = tf.keras.layers.Concatenate()([u2, d1])
    u3 = up_block(u2, 32)
    u3 = tf.keras.layers.Concatenate()([u3, mag_padded])

    # Remove paddings
    x = tf.keras.layers.Lambda(lambda x: x[:,2:-2,:])(u3)
    # Restore shapes
    x = tf.keras.layers.Conv1D(1025, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Multiply Binary Mask with stft
    x = tf.keras.layers.Multiply()([x, mag])

    complex = tf.cast(x, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))

    # Convert back into time domain
    x = tf.signal.inverse_stft(complex, 2048, 512)

    start = (146944 - constants.N_SAMPLES_OUT) // 2
    end = start + constants.N_SAMPLES_OUT
    x = tf.keras.layers.Lambda(lambda x: x[:, start:end])(x)
    x = tf.keras.layers.Reshape((constants.N_SAMPLES_OUT, 1))(x)
    x = tf.keras.layers.Concatenate()([x, wu['vocals']])
    x = tf.keras.layers.Conv1D(1,1, activation='tanh', padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
