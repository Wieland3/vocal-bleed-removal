import sys
import tensorflow as tf
from src import constants
from src.dataset.dataset import DataSet

sys.path.insert(0, constants.WAVE_UNET)
from wave_u_net_tf2.wave_u_net import wave_u_net
from src.train import demucs

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

if __name__ == "__main__":

    USE_ARTIFICIAL = False

    # Load training data
    train = DataSet(subsets="train", use_artificial=USE_ARTIFICIAL)
    tf_dataset_train = train.get_tf_dataset()
    tf_dataset_train = tf_dataset_train.shuffle(buffer_size=4000).batch(16).prefetch(tf.data.AUTOTUNE)

    # Load testing data
    test = DataSet(subsets="test", use_artificial=USE_ARTIFICIAL)
    tf_dataset_test = test.get_tf_dataset()
    tf_dataset_test = tf_dataset_test.batch(16).prefetch(tf.data.AUTOTUNE)

    # Tensorflow checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=constants.CHECKPOINTS_DIR + "/unexploited_difference_layer/cp.ckpt",
                              save_best_only=True,
                              monitor='val_loss',
                              mode='min',
                              verbose=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
    callbacks_list = [cp_callback, early_stopping_callback]

    # Model parameters
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = wave_u_net(**params)
    #model = tf.keras.models.load_model(constants.CHECKPOINTS_DIR + "/full_train_artificial_unexploited/cp.ckpt")
    model.summary()

    # Compile and Train
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.fit(tf_dataset_train, epochs=100, callbacks=callbacks_list, validation_data=tf_dataset_test, steps_per_epoch=3905, validation_steps=2124)


