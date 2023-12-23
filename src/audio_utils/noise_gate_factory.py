from src.audio_utils.time_noise_gate import TimeNoiseGate
from src.audio_utils.spectral_noise_gate import SpectralNoiseGate


class NoiseGateFactory:
    @staticmethod
    def create_noise_gate(noise_gate_type, **kwargs):
        if noise_gate_type == "time":
            return TimeNoiseGate(**kwargs)
        elif noise_gate_type == "spectral":
            strategy = kwargs.get('strategy')
            value = kwargs.get('value')
            if strategy == "percentile":
                return SpectralNoiseGate.from_percentile(value)
            elif strategy == "threshold":
                return SpectralNoiseGate.from_threshold(value)
            else:
                raise ValueError("Strategy must be percentile or threshold.")




