"""Polyphonic pitch detection via Basic Pitch (Spotify)."""

import numpy as np

from models.note import Note

try:
    from basic_pitch.inference import predict
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class PolyphonicDetector:
    """
    Detects multiple simultaneous pitches using Spotify Basic Pitch.

    Returns note events directly (start, end, midi pitch) — bypasses pYIN.
    """

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def detect_from_file(self, filepath: str) -> list[Note]:
        """Run Basic Pitch on an audio file path."""
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError(
                "basic-pitch is required for polyphonic mode. "
                "Run: pip install -r requirements-poly.txt"
            )

        _, _, note_events = predict(filepath)
        return self._events_to_notes(note_events)

    def detect_from_array(self, y: np.ndarray, sr: int, label: str = "poly") -> list[Note]:
        """Run Basic Pitch on an in-memory waveform (writes a temp WAV)."""
        import tempfile
        import os
        from audio.loader import AudioLoader

        loader = AudioLoader(target_sr=sr)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            loader.save_wav(y, tmp_path, sr=sr)
            notes = self.detect_from_file(tmp_path)
            for note in notes:
                note.stem_name = label
            return notes
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _events_to_notes(self, note_events) -> list[Note]:
        notes: list[Note] = []

        for event in note_events:
            # basic_pitch note event: (start, end, pitch, amplitude, bends)
            start, end, pitch, amplitude, *_ = event
            confidence = float(amplitude)
            if confidence < self.min_confidence:
                continue

            midi_pitch = int(round(pitch))
            midi_pitch = max(0, min(127, midi_pitch))

            if not LIBROSA_AVAILABLE:
                note_name = str(midi_pitch)
                frequency = 440.0
            else:
                note_name = librosa.midi_to_note(midi_pitch, unicode=False)
                frequency = float(librosa.midi_to_hz(midi_pitch))

            if end <= start:
                continue

            notes.append(Note(
                midi_pitch=midi_pitch,
                note_name=note_name,
                start_time=float(start),
                end_time=float(end),
                confidence=confidence,
                frequency=frequency,
            ))

        return sorted(notes, key=lambda n: n.start_time)
