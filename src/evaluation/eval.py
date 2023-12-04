import numpy as np
from mir_eval.separation import bss_eval_sources

from src.train import predict
from src.dataset.dataset import MusDataHandler


class Eval:
    def __init__(self):
        self.handler = MusDataHandler(subsets="test", use_artificial=True, infinite=False)

    def evaluate_model(self, exploited):
        sdrs = []
        for i, (mix, vocals) in enumerate(self.handler.data):
            if not exploited:
                mix = mix[:,0]
            vocals = predict.get_ground_truth(vocals[:,0])
            prediction = predict.predict_song(mix, exploited=exploited)[:,0]

            prediction = np.expand_dims(prediction, axis=0)
            vocals = np.expand_dims(vocals, axis=0)

            sdrs.append(bss_eval_sources(vocals, prediction)[0])

        print(sdrs)
        return np.average(sdrs)


e = Eval()
res = e.evaluate_model(True)
print(res)

