import numpy as np

from src.train import predict
from src.dataset.dataset import MusDataHandler
from src.evaluation.metric import l1_loss_db, sdr
from random import uniform


class Eval:
    def __init__(self):
        self.handler = MusDataHandler(subsets="test", use_artificial=True, infinite=False)

    def evaluate_model(self, exploited):
        sdrs = []
        l1 = []

        for i, (mix, vocals) in enumerate(self.handler.data):
            if not exploited:
                mix = mix[:,0]
            vocals = predict.get_ground_truth(vocals[:,0])
            prediction = predict.predict_song(mix, exploited=exploited)[:,0]

            sdrs.append(sdr(vocals, prediction))
            l1.append(l1_loss_db(vocals, prediction))
            print(sdrs)
            print(l1)

        print(sdrs)
        return np.average(sdrs), np.average(l1)

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
        mix[:,1] = np.roll(mix[:,1], 11025)
        return mix



e = Eval()
res = e.evaluate_model(False)
print(res)

