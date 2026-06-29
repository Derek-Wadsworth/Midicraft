import librosa
import numpy as np

class Preprocessor:
    def normalize(self, y: np.ndarray) -> np.ndarray:
        # scale audio so peak amplitude is 1.0
        return y / np.max(np.abs(y))
    
    def trim_silence(self, y: np.ndarray, sr: int) -> np.ndarray:
        # remove silent sections at beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed

    def apply_preemphasis(self, y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        # boost high frequencies by applying a first-order difference equation
        return np.append(y[0], y[1:] - coeff * y[:-1])