import librosa
import numpy as np

class PitchDetector:
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def detect(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # pyin is a pitch detection algorithm that uses the Yin algorithm
        # much more accurate than autocorrelation methods
        # returns: f0 (frequency over time), voiced/unvoiced mask (is a note present?)
        f0, voiced_flag, voiced_prob = librosa.yin(
            y,
            fmin=librosa.note_to_hz('C2'), # lowest note to care about
            fmax=librosa.note_to_hz('C7'), # highest note to care about
            sr=self.sr,
        )
        return f0, voiced_flag

    def hz_to_note_name(self, hz: float) -> str:
        # convert raw frequency to note name
        # e.g. 440.0 -> 'A4'
        if hz <= 0:
            return None
        return librosa.hz_to_note(hz)

    def hz_to_midi(self, hz: float) -> int:
        # convert frequency to MIDI note number (0-127)
        if hz <= 0:
            return None
        return int(librosa.hz_to_midi(hz))