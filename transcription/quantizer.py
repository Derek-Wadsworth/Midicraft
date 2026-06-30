import numpy as np
from models.note import Note


class Quantizer:
    """
    Snaps note timings to the nearest position on a musical grid.

    In music, notes don't start and end at arbitrary times — they align
    to a rhythmic grid based on the tempo. A 16th note at 120 BPM lasts
    exactly 0.125 seconds. A quarter note lasts 0.5 seconds.

    Quantization:
        - Takes detected note times (which may be slightly off the grid)
        - Snaps them to the nearest grid position
        - Produces a musically "tighter" transcription

    This is also essential for Minecraft note block conversion:
        Redstone clocks are grid-based. Every note must align to a tick.

    Musical grid subdivisions:
        1  = quarter notes    (1 per beat)
        2  = eighth notes     (2 per beat)
        4  = sixteenth notes  (4 per beat)  ← default, good for most music
        8  = thirty-second notes
        16 = sixty-fourth notes
    """

    def __init__(self, tempo: float = 120.0, subdivisions: int = 4):
        """
        Args:
            tempo:         Tempo in BPM.
            subdivisions:  Number of grid divisions per beat.
                           4 = sixteenth notes (standard starting point).
        """
        self.tempo = tempo
        self.subdivisions = subdivisions
        self.beat_duration = 60.0 / tempo
        self.grid_size = self.beat_duration / subdivisions

    @property
    def grid_size_ms(self) -> float:
        """Grid resolution in milliseconds."""
        return self.grid_size * 1000

    def snap_to_grid(self, notes: list[Note]) -> list[Note]:
        """
        Snap all note start and end times to the nearest grid position.

        Args:
            notes: List of Note objects with raw detected timings.

        Returns:
            New list of Note objects with quantized timings.
            Original notes are not modified.
        """
        quantized = []
        for note in notes:
            snapped_start = self._snap(note.start_time)
            snapped_end   = self._snap(note.end_time)

            # ensure minimum duration of one grid unit
            # (a note can't have zero or negative duration after snapping)
            if snapped_end <= snapped_start:
                snapped_end = snapped_start + self.grid_size

            quantized.append(Note(
                midi_pitch=note.midi_pitch,
                note_name=note.note_name,
                start_time=round(snapped_start, 6),
                end_time=round(snapped_end, 6),
                confidence=note.confidence,
                frequency=note.frequency,
                voice_id=note.voice_id,
                stem_name=note.stem_name,
            ))

        return quantized

    def _snap(self, t: float) -> float:
        """Round a time value to the nearest grid position."""
        return round(t / self.grid_size) * self.grid_size

    def to_grid_positions(self, notes: list[Note]) -> list[dict]:
        """
        Convert notes to integer grid positions instead of seconds.

        Returns:
            List of dicts: {grid_start, grid_end, grid_duration, note}

        Grid positions are integers starting at 0.
        grid_start=4 means the note starts on the 5th sixteenth note.

        This is the representation we'll use for Minecraft redstone timing:
        each grid position = one redstone tick interval.
        """
        result = []
        for note in notes:
            grid_start = int(round(note.start_time / self.grid_size))
            grid_end   = int(round(note.end_time   / self.grid_size))
            if grid_end <= grid_start:
                grid_end = grid_start + 1
            result.append({
                "grid_start":    grid_start,
                "grid_end":      grid_end,
                "grid_duration": grid_end - grid_start,
                "note":          note,
            })
        return result

    def grid_info(self) -> dict:
        """Return human-readable information about the current grid settings."""
        return {
            "tempo_bpm":         self.tempo,
            "subdivisions":      self.subdivisions,
            "beat_duration_s":   round(self.beat_duration, 4),
            "grid_size_s":       round(self.grid_size, 4),
            "grid_size_ms":      round(self.grid_size_ms, 2),
            "grids_per_bar_4_4": self.subdivisions * 4,
        }
