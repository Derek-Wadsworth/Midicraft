import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class OnsetDetector:
    """
    Detects note onsets (the moment a new note begins) and estimates tempo.

    An onset is characterized by a sudden increase in energy — a "spike"
    in the signal that indicates a new sound event starting.

    Onset detection works by:
        1. Computing a "novelty function" — how much the spectrum changes
           from one frame to the next
        2. Finding peaks in that novelty function
        3. Each peak = one onset (note beginning)
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def detect_onsets(self, y: np.ndarray) -> np.ndarray:
        """
        Detect onset times in the signal.

        Returns:
            Array of onset times in seconds.
            e.g. [0.0, 0.48, 0.96, 1.44, ...]

        These are used to:
          - Anchor where notes begin (complements pitch detection)
          - Split audio at natural musical boundaries
          - Improve quantization accuracy
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=self.sr,
            units='time',       # return seconds, not frame indices
            backtrack=True,     # snap onset to nearest energy dip before peak
                                # (more musically accurate than the raw peak)
        )
        return onset_frames

    def detect_tempo(self, y: np.ndarray) -> float:
        """
        Estimate the tempo of the audio in BPM.

        Uses librosa's beat tracker which:
            1. Computes an onset strength signal
            2. Estimates the period of the pulse (tempo)
            3. Tracks individual beats

        Returns:
            Tempo in beats per minute (BPM).
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
        return float(tempo)

    def detect_beats(self, y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Detect both tempo and individual beat positions.

        Returns:
            tempo:       BPM
            beat_times:  Array of beat times in seconds
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        return float(tempo), beat_times

    def onset_strength(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the onset strength envelope — how much the spectrum
        changes at each frame.

        Useful for visualization: peaks in this signal correspond to
        note beginnings. This is what the beat tracker uses internally.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")
        return librosa.onset.onset_strength(y=y, sr=self.sr)
