import numpy as np
from mir_eval.separation import bss_eval_sources

from src.train import predict
from src.dataset.dataset import MusDataHandler
from src.train.loss import frequency_domain_loss


class Eval:
    def __init__(self):
        self.handler = MusDataHandler(subsets="test", use_artificial=True, infinite=False)

    def evaluate_model(self, exploited):
        sdrs = []
        freq_loss = []

        for i, (mix, vocals) in enumerate(self.handler.data):
            if not exploited:
                mix = mix[:,0]
            vocals = predict.get_ground_truth(vocals[:,0])
            prediction = predict.predict_song(mix, exploited=exploited)[:,0]

            prediction = np.expand_dims(prediction, axis=0)
            vocals = np.expand_dims(vocals, axis=0)

            sdrs.append(bss_eval_sources(vocals, prediction)[0])
            freq_loss.append(frequency_domain_loss(vocals, prediction))

        print(sdrs)
        print(freq_loss)
        return np.average(sdrs), np.average(freq_loss)


e = Eval()
res = e.evaluate_model(False)
print(res)

