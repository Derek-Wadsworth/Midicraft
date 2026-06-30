"""Stem separation configuration and per-stem transcription profiles."""

DEFAULT_STEMS = ("vocals", "bass", "other")
ALL_STEMS = ("vocals", "bass", "other", "drums")

# MIDI channel per stem (channel 9 = GM drums)
STEM_CHANNEL_MAP: dict[str, int] = {
    "vocals": 0,
    "bass": 1,
    "other": 2,
    "drums": 9,
}

# Per-stem pitch-detection tuning (librosa note names for fmin/fmax)
STEM_PROFILES: dict[str, dict] = {
    "vocals": {"fmin": "C3", "fmax": "C7", "min_confidence": 0.4},
    "bass": {"fmin": "E1", "fmax": "G3", "min_confidence": 0.35},
    "other": {"fmin": "C2", "fmax": "C7", "min_confidence": 0.45},
}

DEFAULT_STEM_MODEL = "htdemucs"
DEMUCS_SAMPLE_RATE = 44100
