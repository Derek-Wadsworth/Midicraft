import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class SpectrumAnalyzer:
    """
    Converts raw audio signals into frequency-domain representations.

    This is the mathematical core of the entire pipeline.
    Everything downstream — pitch detection, chord detection, key detection —
    depends on understanding what frequencies are present in the signal.

    Key concepts:
      FFT:   Fast Fourier Transform. Converts a time-domain signal (samples over time)
             into a frequency-domain signal (amplitude at each frequency).
             The result tells you: "at this moment, these frequencies are present."

      STFT:  Short-Time Fourier Transform. Apply FFT repeatedly over sliding
             windows. Produces a 2D array: (frequency_bins × time_frames).
             This is a spectrogram.

      Mel:   A perceptual frequency scale that mimics how humans hear.
             Low frequencies are spaced further apart (we're more sensitive there).
             High frequencies are compressed. Most audio ML models use mel scale.

      Chroma: Collapse all frequency bins into 12 pitch classes (C, C#, D... B),
              summing across all octaves. Captures harmony regardless of register.
    """

    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        """
        Args:
            sr:         Sample rate.
            n_fft:      FFT window size. Larger = better frequency resolution,
                        worse time resolution. 2048 is a standard tradeoff.
            hop_length: Samples between FFT windows. Smaller = more time frames.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # FFT (single window)
    # ------------------------------------------------------------------

    def fft(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the FFT of a single audio frame.

        Returns:
            frequencies: Array of frequency values in Hz for each bin.
            magnitudes:  Amplitude of each frequency bin (0 = absent, high = loud).

        What you're seeing in the output:
            Each bin represents a small range of frequencies.
            Bin width = sr / n_fft  (e.g. 22050 / 2048 ≈ 10.8 Hz per bin)
        """
        # apply a Hann window to reduce spectral leakage
        # (leakage = energy "bleeding" into neighboring frequency bins)
        window = np.hanning(len(frame))
        windowed = frame * window

        # compute FFT — result is complex numbers
        spectrum = np.fft.rfft(windowed, n=self.n_fft)

        # magnitude spectrum — absolute value of complex FFT output
        magnitudes = np.abs(spectrum)

        # corresponding frequency for each bin
        frequencies = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)

        return frequencies, magnitudes

    def fft_db(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same as fft() but returns magnitudes in decibels (log scale)."""
        frequencies, magnitudes = self.fft(frame)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)  # +epsilon avoids log(0)
        return frequencies, magnitudes_db

    # ------------------------------------------------------------------
    # STFT (full signal)
    # ------------------------------------------------------------------

    def stft(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the Short-Time Fourier Transform of the full signal.

        Returns:
            Complex spectrogram of shape (n_fft//2 + 1, n_frames).
            Use stft_magnitude() or stft_db() for the amplitude version.
        """
        if LIBROSA_AVAILABLE:
            return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        else:
            return self._stft_numpy(y)

    def stft_magnitude(self, y: np.ndarray) -> np.ndarray:
        """Return the magnitude (absolute value) of the STFT."""
        return np.abs(self.stft(y))

    def stft_db(self, y: np.ndarray) -> np.ndarray:
        """Return the STFT magnitude in decibels."""
        magnitude = self.stft_magnitude(y)
        if LIBROSA_AVAILABLE:
            return librosa.amplitude_to_db(magnitude, ref=np.max)
        else:
            return 20 * np.log10(magnitude / (np.max(magnitude) + 1e-10) + 1e-10)

    def _stft_numpy(self, y: np.ndarray) -> np.ndarray:
        """Pure numpy STFT fallback when librosa is not available."""
        window = np.hanning(self.n_fft)
        n_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = y[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            stft_matrix[:, i] = np.fft.rfft(frame * window)

        return stft_matrix

    # ------------------------------------------------------------------
    # Mel spectrogram
    # ------------------------------------------------------------------

    def mel_spectrogram(self, y: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Compute the mel-scale spectrogram.

        The mel scale warps the frequency axis to match human hearing:
          - Linear spacing below ~1000 Hz (we're very sensitive here)
          - Logarithmic spacing above ~1000 Hz (our pitch discrimination degrades)

        This is the standard input for most audio deep learning models.

        Returns:
            Array of shape (n_mels, n_frames). Power (amplitude squared) at each
            mel frequency bin over time.
        """
        if LIBROSA_AVAILABLE:
            return librosa.feature.melspectrogram(
                y=y, sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=n_mels
            )
        else:
            raise ImportError("librosa required for mel spectrogram")

    def mel_spectrogram_db(self, y: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Return mel spectrogram in decibels (most useful for visualization and ML)."""
        mel = self.mel_spectrogram(y, n_mels)
        if LIBROSA_AVAILABLE:
            return librosa.power_to_db(mel, ref=np.max)
        return 10 * np.log10(mel + 1e-10)

    # ------------------------------------------------------------------
    # Chroma
    # ------------------------------------------------------------------

    def chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Compute chroma features — energy in each of the 12 pitch classes over time.

        Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B

        Returns:
            Array of shape (12, n_frames). Each row is one pitch class.
            Value = how much energy is in that pitch class at that time.

        Why this is powerful for chord detection:
            A C major chord (C-E-G) will show high energy in rows 0, 4, 7.
            This pattern is the same regardless of octave.
            Chroma is the most direct feature for harmony analysis.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for chroma features")
        return librosa.feature.chroma_stft(
            y=y, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

    def chroma_mean(self, y: np.ndarray) -> np.ndarray:
        """
        Return the average chroma vector over the whole signal.
        Shape: (12,) — one value per pitch class.
        Used for key detection: the dominant pitch classes reveal the key.
        """
        return self.chroma(y).mean(axis=1)

    # ------------------------------------------------------------------
    # MFCCs
    # ------------------------------------------------------------------

    def mfcc(self, y: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute Mel-Frequency Cepstral Coefficients.

        MFCCs capture the timbral texture of sound — what an instrument
        sounds like, independent of pitch.

        How it works:
            1. Compute mel spectrogram
            2. Take the log (mimics human loudness perception)
            3. Apply DCT (decorrelates the features)
            4. Keep only the first n_mfcc coefficients

        The first coefficient (MFCC-0) ≈ overall loudness.
        Coefficients 1-12 capture spectral shape (timbre).

        Returns:
            Array of shape (n_mfcc, n_frames).
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for MFCCs")
        return librosa.feature.mfcc(
            y=y, sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

    # ------------------------------------------------------------------
    # Spectral descriptors (useful for instrument detection)
    # ------------------------------------------------------------------

    def spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """
        The 'center of mass' of the spectrum at each frame.
        High centroid = bright, treble-heavy sound.
        Low centroid = warm, bass-heavy sound.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required")
        return librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)

    def spectral_rolloff(self, y: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
        """
        Frequency below which roll_percent of the total spectral energy lies.
        Useful for distinguishing voiced (tonal) vs unvoiced (noisy) sounds.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required")
        return librosa.feature.spectral_rolloff(y=y, sr=self.sr, roll_percent=roll_percent, hop_length=self.hop_length)

    def zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """
        How often the signal crosses zero per frame.
        High ZCR = noisy, percussive sounds (hi-hats, consonants).
        Low ZCR = tonal sounds (sustained notes, vowels).
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required")
        return librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)

    # ------------------------------------------------------------------
    # Frequency bin utilities
    # ------------------------------------------------------------------

    def frequency_bins(self) -> np.ndarray:
        """Return the center frequency (Hz) of each FFT bin."""
        return np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)

    def bin_for_frequency(self, hz: float) -> int:
        """Return the FFT bin index closest to a given frequency in Hz."""
        bins = self.frequency_bins()
        return int(np.argmin(np.abs(bins - hz)))

    def frame_times(self, n_frames: int) -> np.ndarray:
        """Return the center time (seconds) for each STFT frame."""
        return np.array([i * self.hop_length / self.sr for i in range(n_frames)])
