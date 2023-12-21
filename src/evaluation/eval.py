import numpy as np

from src.train import predict
from src.dataset.dataset import MusDataHandler
from src.evaluation.metric import l1_loss_db, sdr
from src.audio_utils import audio_utils
from src import constants
import soundfile as sf


class Eval:
    def __init__(self):
        self.handler = MusDataHandler(subsets="test", use_artificial=True, infinite=False)

    def evaluate_from_file(self):
        sdrs = []
        l1 = []

        for i, _ in enumerate(self.handler.data):
            pred, _ = sf.read(f"{constants.TRACKS_DIR}/eda/moises_test/left_{i}.mp3")
            pred = audio_utils.stereo_to_mono(pred).squeeze(1)
            vocals, _ = sf.read(f"{constants.TRACKS_DIR}/eda/art_test/left_{i}.mp3")
            vocals = audio_utils.stereo_to_mono(vocals).squeeze(1)
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
            else:
                mix = self.corrupt_clean_sources(mix)
            vocals = predict.get_ground_truth(vocals[:,0])
            prediction = predict.predict_song(mix, exploited=exploited)[:,0]
            audio_utils.save_array_as_wave(prediction,
                                           constants.TRACKS_DIR + f"/eda/art_test/pred_{i}.wav")
            sdrs.append(sdr(vocals, prediction))
            l1.append(l1_loss_db(vocals, prediction))
            print(sdrs)
            print(l1)

        print(sdrs)
        return np.average(sdrs), np.average(l1)

    @staticmethod
    def corrupt_clean_sources(mix):
        noise_level = 0.4
        white_noise = np.random.normal(0, 1, mix.shape[0])

        mix_max = np.max(np.abs(mix[:,1]))
        noise_max = np.max(np.abs(white_noise))
        scaled_noise = white_noise * (mix_max / noise_max) * noise_level

        mix[:,1] = mix[:,1] + scaled_noise
        return mix

    @staticmethod
    def shift_clean_sources(mix):
        mix[:,1] = np.roll(mix[:,1], 44100)
        return mix

    @staticmethod
    def silence_clean_sources(mix):
        mix[:,1] = np.zeros(mix[:,1].shape)
        return mix


e = Eval()
res = e.evaluate_from_file()
print(res)


