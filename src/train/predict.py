import tensorflow as tf
import numpy as np
from src import constants
import soundfile as sf
from src.dataset import dataset
from src.audio_utils import audio_utils
import sys

sys.path.insert(0, constants.WAVE_UNET)
from wave_u_net import wave_u_net

params = {
  "num_initial_filters": 12,
  "num_layers": 12,
  "kernel_size": 15,
  "merge_filter_size": 5,
  "source_names": ["vocals"],
  "num_channels": 2,
  "output_filter_size": 1,
  "padding": "valid",
  "input_size": 147443,
  "context": True,
  "upsampling_type": "learned",         # "learned" or "linear"
  "output_activation": "linear",        # "linear" or "tanh"
  "output_type": "direct",          # "direct" or "difference"
}

if __name__ == "__main__":
    test = dataset.DataSet(subsets="test")

    checkpoint_path = constants.CHECKPOINTS_DIR + "/best_model/cp.ckpt"
    model = tf.keras.models.load_model(checkpoint_path)

    """
    song_index = 2
    song = test.songs[song_index]
    vocals = test.vocals[song_index]
    """
    pred = []
    gt = []

    song, sr = sf.read(constants.TRACKS_DIR + "/thomas/voice.wav")
    vocals, sr = sf.read(constants.TRACKS_DIR + "/thomas/voice.wav")

    song = np.stack([song, song], axis=1)
    vocals = np.stack([vocals, vocals], axis=1)

    for i, (X_chunk, y_chunk) in enumerate(test.song_data_generator(song, vocals)):
        X_chunk_batch = np.expand_dims(X_chunk, axis=0)
        y_pred_chunk = model.predict(X_chunk_batch)['vocals'].squeeze(0)
        pred.append(y_pred_chunk)
        gt.append(y_chunk)

    pred = np.concatenate(pred, axis=0)
    gt = np.concatenate(gt, axis=0)

    audio_utils.save_array_as_wave(pred, constants.DEBUGGING_DATA_DIR + "/PRED1.wav")
    audio_utils.save_array_as_wave(gt, constants.DEBUGGING_DATA_DIR + "/GT1.wav")






