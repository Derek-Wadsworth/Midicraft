from dataclasses import dataclass, field

import numpy as np

from models.note import Note


@dataclass
class TranscriptionResult:
  """Output of transcribing a single audio source (mix or one stem)."""

  notes: list[Note]
  tempo: float
  source_label: str = "mix"
  f0: np.ndarray | None = None
  voiced_flag: np.ndarray | None = None
  voiced_prob: np.ndarray | None = None
  times: np.ndarray | None = None
  y: np.ndarray | None = None
  sr: int = 22050


@dataclass
class MergedTranscriptionResult:
  """Combined output from multi-stem or polyphonic transcription."""

  notes: list[Note]
  tempo: float
  stem_results: list[TranscriptionResult] = field(default_factory=list)
  y: np.ndarray | None = None
  sr: int = 22050
  f0: np.ndarray | None = None
  voiced_flag: np.ndarray | None = None
