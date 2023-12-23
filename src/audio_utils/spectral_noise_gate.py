from src.audio_utils.noise_gate import NoiseGate
import numpy as np
import librosa


class SpectralNoiseGate(NoiseGate):
    def __init__(self, threshold, percentile):
        super().__init__(threshold)
        self.percentile = percentile

    @classmethod
    def from_threshold(cls, threshold):
        return cls(threshold, None)

    @classmethod
    def from_percentile(cls, percentile):
        return cls(None, percentile)

    def process(self, audio):
        if self.percentile:
            return self.process_percentile(audio)
        elif self.threshold:
            return self.process_threshold(audio)

    def process_percentile(self, audio):
        stft = librosa.stft(audio)
        mag = np.abs(stft)
        lower = np.percentile(mag, 100-self.percentile)
        upper = np.percentile(mag, self.percentile)
        stft_filtered = np.where((mag > upper), mag, 0)
        stft_filtered = stft_filtered * np.exp(1j * np.angle(stft))
        return librosa.istft(stft_filtered)

    def process_threshold(self, audio):
        stft = librosa.stft(audio)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=1)
        stft_db_filtered = np.where((stft_db > self.threshold), stft_db, -100)
        stft_filtered = librosa.db_to_amplitude(stft_db_filtered) * np.exp(1j * np.angle(stft))
        return librosa.istft(stft_filtered)






