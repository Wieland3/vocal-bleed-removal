import sys
from src import constants

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
    m = wave_u_net(**params)
    m.summary()
