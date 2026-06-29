import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class Segmenter:
    """
    Splits an audio signal into smaller pieces for analysis.

    Why do we segment audio?
      - Feature extraction (FFT, chroma) works on short windows, not whole files
      - Different segments can be analyzed in parallel
      - Silence can be removed to speed up processing
      - Songs can be split into sections (verse, chorus) for section-level analysis

    Key concepts:
      - Frame:  A short window of samples (e.g. 2048 samples ≈ 93ms at 22050 Hz)
      - Hop:    How far we move between frames (e.g. 512 samples ≈ 23ms)
      - Overlap: frame_size - hop_size (frames overlap to avoid missing transients)
    """

    def __init__(self, sr: int = 22050, frame_size: int = 2048, hop_size: int = 512):
        """
        Args:
            sr:         Sample rate of the audio signal.
            frame_size: Number of samples per frame (window size for FFT).
                        2048 is standard — balances time and frequency resolution.
            hop_size:   Samples between consecutive frames.
                        512 = 75% overlap with frame_size 2048. More overlap = smoother features.
        """
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    # ------------------------------------------------------------------
    # Fixed-size windowing
    # ------------------------------------------------------------------

    def to_frames(self, y: np.ndarray) -> np.ndarray:
        """
        Split signal into overlapping frames.

        Returns:
            Array of shape (n_frames, frame_size).
            Each row is one frame (window) of audio samples.

        This is what librosa does internally before computing FFT.
        Understanding this helps you debug feature extraction.
        """
        n_frames = 1 + (len(y) - self.frame_size) // self.hop_size
        frames = np.zeros((n_frames, self.frame_size))

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frames[i] = y[start:end]

        return frames

    def frame_times(self, n_frames: int) -> np.ndarray:
        """
        Return the center time (in seconds) of each frame.
        Used to align features back to real time.
        """
        return np.array([
            (i * self.hop_size + self.frame_size // 2) / self.sr
            for i in range(n_frames)
        ])

    # ------------------------------------------------------------------
    # Silence removal
    # ------------------------------------------------------------------

    def remove_silence(
        self,
        y: np.ndarray,
        threshold_db: float = -40.0,
        min_duration_s: float = 0.1
    ) -> list[tuple[np.ndarray, float]]:
        """
        Split audio into non-silent segments, discarding quiet gaps.

        Args:
            threshold_db:   Sections quieter than this (in dB) are treated as silence.
                            -40 dB is a good default. Lower = more aggressive trimming.
            min_duration_s: Discard segments shorter than this (likely noise/artifacts).

        Returns:
            List of (segment_audio, start_time_seconds) tuples.
            Each segment is a contiguous non-silent region.

        Why this matters:
            Silence confuses pitch detectors. Removing it improves accuracy
            and speeds up processing significantly.
        """
        if not LIBROSA_AVAILABLE:
            # fallback: energy-based simple silence detection
            return self._energy_silence_removal(y, min_duration_s)

        # librosa gives us intervals of non-silent audio
        intervals = librosa.effects.split(y, top_db=abs(threshold_db))

        min_samples = int(min_duration_s * self.sr)
        segments = []

        for start_sample, end_sample in intervals:
            if (end_sample - start_sample) < min_samples:
                continue  # skip very short blips
            segment = y[start_sample:end_sample]
            start_time = start_sample / self.sr
            segments.append((segment, start_time))

        return segments

    def _energy_silence_removal(self, y: np.ndarray, min_duration_s: float) -> list:
        """Fallback silence removal using RMS energy when librosa is unavailable."""
        frame_energy = np.array([
            np.sqrt(np.mean(y[i:i+self.frame_size]**2))
            for i in range(0, len(y) - self.frame_size, self.hop_size)
        ])
        threshold = np.max(frame_energy) * 0.01  # 1% of peak energy
        voiced = frame_energy > threshold
        segments = []
        in_segment = False
        start_frame = 0

        for i, v in enumerate(voiced):
            if v and not in_segment:
                in_segment = True
                start_frame = i
            elif not v and in_segment:
                in_segment = False
                start_sample = start_frame * self.hop_size
                end_sample = i * self.hop_size
                if (end_sample - start_sample) >= int(min_duration_s * self.sr):
                    segments.append((y[start_sample:end_sample], start_sample / self.sr))

        return segments

    # ------------------------------------------------------------------
    # Fixed-duration chunking
    # ------------------------------------------------------------------

    def to_chunks(self, y: np.ndarray, chunk_duration_s: float = 5.0) -> list[tuple[np.ndarray, float]]:
        """
        Split audio into equal-length chunks.

        Useful for:
          - Processing long files in batches (memory efficiency)
          - Real-time streaming simulation
          - Feeding a model that expects fixed-length input

        Args:
            chunk_duration_s: Length of each chunk in seconds.

        Returns:
            List of (chunk_audio, start_time_seconds) tuples.
            Last chunk may be shorter than chunk_duration_s.
        """
        chunk_size = int(chunk_duration_s * self.sr)
        chunks = []

        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            start_time = i / self.sr
            chunks.append((chunk, start_time))

        return chunks

    # ------------------------------------------------------------------
    # Beat-aligned segmentation
    # ------------------------------------------------------------------

    def to_beats(self, y: np.ndarray) -> tuple[list[tuple[np.ndarray, float]], float]:
        """
        Split audio into beat-aligned segments.

        Instead of fixed time windows, segments follow the musical beat.
        This produces musically meaningful segments — each chunk is one beat.

        Returns:
            - List of (beat_audio, start_time_seconds) tuples
            - Detected tempo in BPM

        Why this matters for Minecraft:
            Redstone timing is grid-based. Beat-aligned segments map
            naturally onto the redstone clock grid.
        """
        if not LIBROSA_AVAILABLE:
            print("[Segmenter] librosa required for beat tracking.")
            return [], 120.0

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
        beat_samples = librosa.frames_to_samples(beat_frames)
        tempo = float(tempo)

        segments = []
        for i in range(len(beat_samples) - 1):
            start = beat_samples[i]
            end = beat_samples[i + 1]
            segment = y[start:end]
            start_time = start / self.sr
            segments.append((segment, start_time))

        return segments, tempo

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, y: np.ndarray) -> dict:
        """Return basic information about the signal and segmentation parameters."""
        duration = len(y) / self.sr
        n_frames = 1 + (len(y) - self.frame_size) // self.hop_size
        frame_duration_ms = self.frame_size / self.sr * 1000
        hop_duration_ms = self.hop_size / self.sr * 1000

        return {
            "signal_length_samples": len(y),
            "duration_seconds": round(duration, 3),
            "sample_rate": self.sr,
            "frame_size_samples": self.frame_size,
            "hop_size_samples": self.hop_size,
            "frame_duration_ms": round(frame_duration_ms, 2),
            "hop_duration_ms": round(hop_duration_ms, 2),
            "overlap_percent": round((1 - self.hop_size / self.frame_size) * 100, 1),
            "n_frames": n_frames,
        }
