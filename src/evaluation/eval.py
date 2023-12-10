import numpy as np

from src.train import predict
from src.dataset.dataset import MusDataHandler
from src.evaluation.metric import frequency_domain_loss, sdr


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

            print(sdr(vocals, prediction))
            sdrs.append(sdr(vocals, prediction))

        print(sdrs)
        return np.average(sdrs)


e = Eval()
res = e.evaluate_model(False)
print(res)

