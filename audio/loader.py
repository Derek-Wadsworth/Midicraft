import librosa
import numpy as np

class AudioLoader:
    def __init__(sel, target_sr=22050):
        self.target_sr = target_sr # standard sample rate for audio processing

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        # librosa handles mp3, wav, flac, ogg automatically
        # mono = true converts stereo to mono
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        return y, sr
    
    def load_from_microphone(self, duration: int = 5):
        # future implementation for microphone input
        pass