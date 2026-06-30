"""Stem separation via Demucs (htdemucs)."""

import os
from typing import Callable

import numpy as np

from separation.stem_config import (
    ALL_STEMS,
    DEMUCS_SAMPLE_RATE,
    DEFAULT_STEM_MODEL,
)

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class StemSeparator:
    """
    Separates a mixed audio file into stems using Demucs.

    Returns mono float32 numpy arrays per stem at DEMUCS_SAMPLE_RATE (44100 Hz).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_STEM_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or self._default_device()
        self._model = None

    @staticmethod
    def _default_device() -> str:
        if not DEMUCS_AVAILABLE:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if not DEMUCS_AVAILABLE:
            raise ImportError(
                "demucs and torch are required for stem separation. "
                "Run: pip install -r requirements-stems.txt"
            )
        if self._model is None:
            print(f"[StemSeparator] Loading model '{self.model_name}' on {self.device}...")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def separate(
        self,
        filepath: str,
        stems: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, tuple[np.ndarray, int]]:
        """
        Separate audio into stems.

        Args:
            filepath: Path to input audio.
            stems:    Subset of stem names to return (default: all four).

        Returns:
            dict mapping stem name -> (mono waveform, sample_rate).
        """
        self._load_model()
        requested = stems or list(ALL_STEMS)

        if progress_callback:
            progress_callback(f"Loading audio for separation: {filepath}")

        wav, sr = self._load_audio_stereo(filepath)

        if progress_callback:
            progress_callback(f"Running Demucs on {self.device} (this may take several minutes on CPU)...")

        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav.unsqueeze(0).to(self.device),
                device=self.device,
                progress=True,
            )[0]

        # htdemucs source order matches model.sources
        source_names = self._model.sources
        result: dict[str, tuple[np.ndarray, int]] = {}

        for name, source in zip(source_names, sources):
            if name not in requested:
                continue
            mono = source.mean(dim=0).cpu().numpy().astype(np.float32)
            peak = np.max(np.abs(mono))
            if peak > 0:
                mono = mono / peak
            result[name] = (mono, sr)

        if progress_callback:
            progress_callback(f"Separated {len(result)} stems: {', '.join(result.keys())}")

        return result

    def _load_audio_stereo(self, filepath: str) -> tuple["torch.Tensor", int]:
        """
        Load stereo audio via librosa (avoids torchaudio/torchcodec version issues).
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for stem separation audio loading.")

        y, sr = librosa.load(filepath, sr=None, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])

        if sr != DEMUCS_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=DEMUCS_SAMPLE_RATE)
            sr = DEMUCS_SAMPLE_RATE

        wav = torch.from_numpy(y.astype(np.float32))
        return wav, sr

    def separate_and_save(
        self,
        filepath: str,
        output_dir: str,
        stems: list[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """
        Separate audio and write stem WAV files.

        Returns:
            dict mapping stem name -> path to saved WAV.
        """
        from audio.loader import AudioLoader

        separated = self.separate(filepath, stems=stems, progress_callback=progress_callback)
        os.makedirs(output_dir, exist_ok=True)

        paths: dict[str, str] = {}
        loader = AudioLoader()
        for name, (y, sr) in separated.items():
            path = os.path.join(output_dir, f"{name}.wav")
            loader.save_wav(y, path, sr=sr)
            paths[name] = path
            print(f"[StemSeparator] Saved {name} -> {path}")

        return paths

    @staticmethod
    def load_existing_stems(stems_dir: str, stems: list[str] | None = None) -> dict[str, tuple[np.ndarray, int]]:
        """Load pre-separated stem WAVs from a directory."""
        from audio.loader import AudioLoader

        loader = AudioLoader(target_sr=22050)
        requested = stems or list(ALL_STEMS)
        result: dict[str, tuple[np.ndarray, int]] = {}

        for name in requested:
            path = os.path.join(stems_dir, f"{name}.wav")
            if not os.path.isfile(path):
                continue
            y, sr = loader.load(path)
            result[name] = (y, sr)

        if not result:
            raise FileNotFoundError(f"No stem WAV files found in {stems_dir}")

        return result
