from matplotlib.colors import BoundaryNorm, ListedColormap
from src.inference import predict
from src.dataset.dataset import MusdbDataHandler
from src.evaluation.metric import l1_loss_db, sdr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


class Eval:
    def __init__(self, exploited=False):
        self.exploited = exploited
        self.handler = MusdbDataHandler(subsets="test", use_artificial=True, infinite=False, exploited=self.exploited)

    def evaluate_model(self):
        sdrs = []
        l1 = []

        for i, (mix, vocals) in enumerate(self.handler.data):
            vocals = predict.get_ground_truth(vocals[:,0])
            prediction = predict.predict_song(mix, exploited=self.exploited)[:,0]
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
    def plot_p_value_matrix(data_arrays, method_names, reference=None, reference_name=None):
        num_methods = len(method_names)
        p_values_matrix = np.zeros((num_methods, num_methods)) if not reference else np.zeros((num_methods, 1))

        for i in range(num_methods):

            if reference:
                _, p_value = ttest_rel(data_arrays[i], reference)
                p_values_matrix[i] = p_value
                continue

            for j in range(i, num_methods):
                if reference:
                    break
                _, p_value = ttest_rel(data_arrays[i], data_arrays[j])
                p_values_matrix[i, j] = p_value
                p_values_matrix[j, i] = p_value

        plt.figure(figsize=(10, 8))

        my_colors = ['#DCFFB7', '#FF6868']
        my_cmap = ListedColormap(my_colors)
        bounds = [0, 0.05, 1]
        my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

        x_labels = method_names if not reference else reference_name
        ax = sns.heatmap(p_values_matrix, annot=True, norm=my_norm, cmap=my_cmap, xticklabels=x_labels,
                         yticklabels=method_names, annot_kws={"fontsize": 15}, vmin=0, vmax=1, cbar=False)
        ax.figure.tight_layout()
        plt.show()

"""
e = Eval(True)
res = e.evaluate_model()
print("SDR", res[0])
print("L1", res[1])
"""



