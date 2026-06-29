import librosa
import numpy as np

class OnsetDetector:
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def detct_onsets(self, y: np.ndarray) -> np.ndarray:
        # onset = moment a new note starts (spike in amplitude)
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=self.sr,
            units='time',
        )
        return onset_frames

    def detect_tempo(self, y: np.ndarray) -> float:
        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
        return float(tempo) #bpm

