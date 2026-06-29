import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class PitchDetector:
    """
    Detects the fundamental frequency (pitch) of a monophonic audio signal.

    Algorithm used: pYIN (probabilistic YIN)
        - An extension of the YIN algorithm (de Cheveigné & Kawahara, 2002)
        - Uses a Hidden Markov Model to smooth pitch estimates over time
        - Much more accurate than basic autocorrelation
        - Returns a confidence score per frame (voiced probability)

    YIN works by:
        1. Computing the difference function: how much does the signal differ
           from a time-shifted copy of itself?
        2. The time shift where the difference is minimized = the period (1/frequency)
        3. Invert the period to get the frequency
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def detect(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch (f0) across the full signal.

        Returns:
            f0:          Fundamental frequency in Hz at each frame.
                         0.0 where no pitch is detected (unvoiced frames).
            voiced_flag: Boolean array — True where a pitched note is detected.

        Frame rate:
            By default librosa uses hop_length=512 samples.
            At 22050 Hz: 512/22050 ≈ 23ms per frame.
            A 3-second clip → ~130 frames.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for pitch detection.")

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz  — below bass guitar low E
            fmax=librosa.note_to_hz('C7'),   # ~2093 Hz — above soprano high C
            sr=self.sr,
        )

        # pyin returns NaN for unvoiced frames — replace with 0.0
        f0 = np.nan_to_num(f0, nan=0.0)

        return f0, voiced_flag

    def detect_with_confidence(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Like detect() but also returns the voiced probability per frame.

        voiced_prob values:
            ~1.0 = very confident a note is present
            ~0.5 = uncertain
            ~0.0 = confident there is no note (silence/noise)

        Use this to filter out low-confidence detections.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0, voiced_flag, voiced_prob

    def hz_to_note_name(self, hz: float) -> str | None:
        """Convert frequency in Hz to note name, e.g. 440.0 → 'A4'."""
        if not LIBROSA_AVAILABLE or hz <= 0:
            return None
        return librosa.hz_to_note(hz)

    def hz_to_midi(self, hz: float) -> int | None:
        """Convert frequency in Hz to MIDI note number, e.g. 440.0 → 69."""
        if not LIBROSA_AVAILABLE or hz <= 0:
            return None
        return int(round(librosa.hz_to_midi(hz)))

    def midi_to_hz(self, midi: int) -> float:
        """Convert MIDI note number to frequency in Hz, e.g. 69 → 440.0."""
        if not LIBROSA_AVAILABLE:
            return 440.0 * (2 ** ((midi - 69) / 12))
        return float(librosa.midi_to_hz(midi))