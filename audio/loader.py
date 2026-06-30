import os

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioLoader:
    """
    Loads audio files from disk into numpy arrays.

    Handles: .mp3, .wav, .flac, .ogg, .aiff, .m4a (via librosa/soundfile).
    Always outputs: mono float32 numpy array, normalized to [-1.0, 1.0].
    """

    def __init__(self, target_sr: int = 22050):
        """
        Args:
            target_sr: Sample rate to resample to on load.
                       22050 Hz is the standard for music analysis.
                       (Half of CD quality 44100 Hz — sufficient for pitch up to 11025 Hz,
                       which covers all musical notes.)
        """
        self.target_sr = target_sr

    def load(self, filepath: str) -> tuple[np.ndarray, int]:
        """
        Load an audio file and return (signal, sample_rate).

        Args:
            filepath: Path to audio file.

        Returns:
            y:  1D numpy array of audio samples, dtype float32, range [-1.0, 1.0].
            sr: Sample rate (always self.target_sr after resampling).
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required. Run: pip install librosa")

        # mono=True: average stereo channels into one (stereo confuses pitch detectors)
        # sr=target_sr: resample to our target rate on load (handles any input rate)
        y, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return y, sr

    def load_segment(self, filepath: str, start_s: float, duration_s: float) -> tuple[np.ndarray, int]:
        """
        Load only a segment of an audio file (avoids loading the whole file).
        Useful for previewing or processing long files in chunks.

        Args:
            filepath:   Path to audio file.
            start_s:    Start time in seconds.
            duration_s: Duration of segment in seconds.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        y, sr = librosa.load(
            filepath,
            sr=self.target_sr,
            mono=True,
            offset=start_s,
            duration=duration_s,
        )
        return y, sr

    def get_duration(self, filepath: str) -> float:
        """Return the duration of an audio file in seconds without fully loading it."""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")
        return librosa.get_duration(path=filepath)

    def get_info(self, filepath: str) -> dict:
        """Return metadata about an audio file."""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")
        duration = self.get_duration(filepath)
        return {
            "filepath": filepath,
            "duration_s": round(duration, 3),
            "target_sr": self.target_sr,
            "n_samples_after_load": int(duration * self.target_sr),
        }

    def save_wav(self, y: np.ndarray, filepath: str, sr: int | None = None) -> str:
        """Save a mono waveform to a WAV file."""
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile is required to save WAV files.") from exc

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        sf.write(filepath, y, sr or self.target_sr)
        return filepath
