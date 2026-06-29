import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class Preprocessor:
    """
    Prepares a raw audio signal for feature extraction and pitch detection.

    Steps applied (in order):
        1. Normalize   — scale amplitude to [-1.0, 1.0]
        2. Trim        — remove leading/trailing silence
        3. Preemphasis — boost high frequencies for better pitch detection
    """

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """
        Scale the signal so the peak amplitude is exactly 1.0.

        Why: different recordings have different loudness levels.
        Normalizing ensures consistent behavior from downstream algorithms
        regardless of how quietly or loudly the source was recorded.
        """
        peak = np.max(np.abs(y))
        if peak == 0:
            return y  # avoid division by zero on silent audio
        return y / peak

    def trim_silence(self, y: np.ndarray, sr: int, top_db: float = 20.0) -> np.ndarray:
        """
        Remove silence from the beginning and end of the signal.

        Args:
            top_db: Sections quieter than (peak - top_db) dB are trimmed.
                    20 dB is a good default. Increase to trim more aggressively.

        Why: Leading/trailing silence wastes processing time and can confuse
             onset detection by adding false "quiet period" context.
        """
        if not LIBROSA_AVAILABLE:
            return self._trim_energy(y)

        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed

    def _trim_energy(self, y: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Fallback energy-based trim when librosa is unavailable."""
        energy = np.abs(y)
        start = np.argmax(energy > threshold)
        end = len(y) - np.argmax(energy[::-1] > threshold)
        return y[start:end] if end > start else y

    def apply_preemphasis(self, y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Apply a preemphasis filter to boost high frequencies.

        Formula: y'[n] = y[n] - coeff * y[n-1]

        This is a first-order high-pass filter. It amplifies high frequencies
        relative to low frequencies, which:
          - Improves pitch detection accuracy for high-pitched instruments
          - Balances the spectrum (low frequencies are naturally louder)
          - Reduces the effect of low-frequency noise and room rumble

        Args:
            coeff: Filter coefficient. 0.97 is standard. Higher = more boost.
                   0.0 = no effect. 1.0 = maximum boost (removes DC component).
        """
        return np.append(y[0], y[1:] - coeff * y[:-1])

    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Run all preprocessing steps in the recommended order."""
        y = self.normalize(y)
        y = self.trim_silence(y, sr)
        y = self.apply_preemphasis(y)
        return y
