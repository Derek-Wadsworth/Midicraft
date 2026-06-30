"""
Maps MIDI pitches to Minecraft note block instrument + note value (0–24).

Minecraft note blocks play F#3 at note value 0; each step raises pitch by one semitone.
Valid range: MIDI 54 (F#3) through MIDI 78 (F#5) → block notes 0–24.
"""

from models.note import Note

# Default instrument per stem / voice lane (matches separation/stem_config voices)
STEM_INSTRUMENTS: dict[str, str] = {
    "vocals": "harp",
    "bass": "bass",
    "other": "guitar",
    "drums": "basedrum",
    "mix": "harp",
    "poly": "harp",
}

# All valid Minecraft 1.20+ note block instruments (lowercase for JSON)
VALID_INSTRUMENTS = frozenset({
    "harp", "bass", "basedrum", "snare", "hat", "guitar", "flute", "bell",
    "chime", "xylophone", "iron_xylophone", "cow_bell", "didgeridoo", "bit",
    "banjo", "pling",
})

# F#3 = MIDI 54 is note block value 0
MIDI_BASE = 54
BLOCK_NOTE_MIN = 0
BLOCK_NOTE_MAX = 24


class NoteBlockMapper:
    """Convert MIDI note numbers to Minecraft note block parameters."""

    def __init__(self, default_instrument: str = "harp"):
        if default_instrument not in VALID_INSTRUMENTS:
            raise ValueError(f"Unknown instrument: {default_instrument}")
        self.default_instrument = default_instrument

    @staticmethod
    def midi_to_block_note(midi_pitch: int) -> int:
        """Map MIDI pitch to note block value 0–24 (clamped)."""
        return max(BLOCK_NOTE_MIN, min(BLOCK_NOTE_MAX, midi_pitch - MIDI_BASE))

    def instrument_for_note(self, note: Note) -> str:
        """Pick instrument from stem name, falling back to default."""
        if note.stem_name and note.stem_name in STEM_INSTRUMENTS:
            return STEM_INSTRUMENTS[note.stem_name]
        return self.default_instrument

    def map_note(self, note: Note) -> dict:
        """Return block mapping fields for a single Note."""
        return {
            "midi_pitch": note.midi_pitch,
            "note_name": note.note_name,
            "block_note": self.midi_to_block_note(note.midi_pitch),
            "block_instrument": self.instrument_for_note(note),
            "confidence": round(note.confidence, 4),
        }
