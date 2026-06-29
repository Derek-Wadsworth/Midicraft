from dataclasses import dataclass, field


@dataclass
class Note:
    """
    Represents a single musical note detected from audio.

    Attributes:
        midi_pitch:  MIDI note number (0-127). Middle C = 60, A4 = 69.
        note_name:   Human-readable name e.g. "A4", "C#3".
        start_time:  Time in seconds when the note begins.
        end_time:    Time in seconds when the note ends.
        confidence:  How confident the detector is (0.0 - 1.0).
        frequency:   Raw detected frequency in Hz (before rounding to MIDI).
    """
    midi_pitch: int
    note_name: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    frequency: float = 0.0

    @property
    def duration(self) -> float:
        """Duration of the note in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Duration of the note in milliseconds."""
        return self.duration * 1000

    def is_valid(self) -> bool:
        """Return True if this note has a positive duration and valid pitch."""
        return self.duration > 0 and 0 <= self.midi_pitch <= 127

    def __repr__(self) -> str:
        return (
            f"Note({self.note_name}, "
            f"start={self.start_time:.3f}s, "
            f"end={self.end_time:.3f}s, "
            f"duration={self.duration:.3f}s, "
            f"confidence={self.confidence:.2f})"
        )
