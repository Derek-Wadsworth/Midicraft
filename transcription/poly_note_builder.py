"""Adapter: polyphonic detector output -> cleaned Note list."""

from models.note import Note
from features.polyphonic import PolyphonicDetector


class PolyNoteBuilder:
    """
    Builds Note lists from polyphonic detectors (Basic Pitch).

    Unlike NoteBuilder, this does not use frame-level FSM — events arrive
    as complete notes from the model.
    """

    def __init__(self, min_confidence: float = 0.4):
        self.detector = PolyphonicDetector(min_confidence=min_confidence)

    def build_from_file(self, filepath: str, label: str = "poly") -> list[Note]:
        notes = self.detector.detect_from_file(filepath)
        for note in notes:
            note.stem_name = label
        return notes

    def build_from_array(self, y, sr: int, label: str = "poly") -> list[Note]:
        return self.detector.detect_from_array(y, sr, label=label)
