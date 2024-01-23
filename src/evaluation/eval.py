import librosa.util
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.train import predict
from src.dataset.dataset import MusDataHandler
from src.evaluation.metric import l1_loss_db, sdr
from src.audio_utils import audio_utils
from src import constants
import soundfile as sf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


class Eval:
    def __init__(self):
        self.handler = MusDataHandler(subsets="test", use_artificial=True, infinite=False)

    def evaluate_from_file(self):
        sdrs = []
        l1 = []

        for i, _ in enumerate(self.handler.data):
            pred, _ = sf.read(f"{constants.TRACKS_DIR}/eda/moises_test/left_{i}.mp3")
            pred = audio_utils.stereo_to_mono(pred).squeeze(1)
            vocals, _ = sf.read(f"{constants.TRACKS_DIR}/eda/art_test/voc_{i}.wav")
            vocals = audio_utils.stereo_to_mono(vocals).squeeze(1)
            vocals = vocals[:pred.shape[0]]
            sdrs.append(sdr(vocals, pred))
            l1.append(l1_loss_db(vocals, pred))
            print(sdrs)
            print(l1)

        return np.average(sdrs), np.average(l1)

    def evaluate_model(self, exploited):
        sdrs = []
        l1 = []

        for i, (mix, vocals) in enumerate(self.handler.data):
            if not exploited:
                mix = mix[:,0]
            vocals = predict.get_ground_truth(vocals[:,0])
            #vocals = vocals[:,0]
            mix = self.silence_clean_sources(mix)
            prediction = predict.predict_song(mix, exploited=exploited)[:,0]
            #noise_gate = NoiseGateFactory().create_noise_gate("time", threshold=-40)
            #prediction = noise_gate.process(mix)
            sdrs.append(sdr(vocals, prediction))
            l1.append(l1_loss_db(vocals, prediction))
            print(sdrs)
            print(l1)

        return sdrs, l1

    @staticmethod
    def corrupt_clean_sources(mix):
        noise_level = 0.2
        white_noise = np.random.normal(0, 1, mix.shape[0])

        mix_max = np.max(np.abs(mix[:,1]))
        noise_max = np.max(np.abs(white_noise))
        scaled_noise = white_noise * (mix_max / noise_max) * noise_level

        mix[:,1] = mix[:,1] + scaled_noise
        return mix

    @staticmethod
    def shift_clean_sources(mix):
        mix[:,1] = np.roll(mix[:,1], 22050)
        return mix

    @staticmethod
    def silence_clean_sources(mix):
        mix[:,1] = np.zeros(mix[:,1].shape)
        return mix
    @staticmethod
    def plot_p_value_matrix(data_arrays, method_names):
        num_methods = len(method_names)
        p_values_matrix = np.zeros((num_methods, num_methods))

        for i in range(num_methods):
            for j in range(i, num_methods):
                _, p_value = ttest_rel(data_arrays[i], data_arrays[j])
                p_values_matrix[i, j] = p_value
                p_values_matrix[j, i] = p_value

        plt.figure(figsize=(10, 8))

        my_colors = ['#DCFFB7', '#FF6868']
        my_cmap = ListedColormap(my_colors)
        bounds = [0, 0.05, 1]
        my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

        ax = sns.heatmap(p_values_matrix, annot=True, norm=my_norm, cmap=my_cmap, xticklabels=method_names,
                         yticklabels=method_names, annot_kws={"fontsize": 15}, vmin=0, vmax=1, cbar=False)
        ax.figure.tight_layout()
        plt.show()

"""
e = Eval()
res = e.evaluate_model(True)
print("SDR", res[0])
print("L1", res[1])
"""




