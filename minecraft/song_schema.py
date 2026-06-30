"""Grid-native song schema for Minecraft redstone sequencers."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GridNoteEvent:
    """One note aligned to the redstone clock grid."""

    grid_start: int
    grid_duration: int
    midi_pitch: int
    note_name: str
    block_note: int
    block_instrument: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SongTrack:
    """A single voice lane (one row of note blocks in the build)."""

    name: str
    voice_id: int
    instrument: str
    events: list[GridNoteEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "voice_id": self.voice_id,
            "instrument": self.instrument,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class MinecraftSong:
    """
    Complete grid-native song for the Minecraft mod.

    Each grid position = one step on the redstone sequencer clock.
    """

    tempo_bpm: float
    subdivisions: int
    grid_size_seconds: float
    ticks_per_beat: int
    total_grid_length: int
    tracks: list[SongTrack] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "midicraft_song_v1",
            "tempo_bpm": round(self.tempo_bpm, 2),
            "subdivisions": self.subdivisions,
            "grid_size_seconds": round(self.grid_size_seconds, 6),
            "ticks_per_beat": self.ticks_per_beat,
            "total_grid_length": self.total_grid_length,
            "track_count": len(self.tracks),
            "tracks": [t.to_dict() for t in self.tracks],
        }
