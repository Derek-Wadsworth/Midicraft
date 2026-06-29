import numpy as np
from models.note import Note

class Quantizer:
    def __init__(self, tempo: float, subdivisions: int = 4):
        self.tempo = tempo
        self.subdivisions = subdivisions # 4 = sixteenth notes
        self.beat_duration = 60.0 / tempo # seconds per beat
        self.grid_size = self.beat_duration / subdivisions # seconds per grid

    def snap_to_grid(self, notes: list[Note]) -> list[Note]:
        # snap each note's start/end to nearest grid boundary
        # this is what a DAW does when you enable "quantize"
        quantized = []
        for note in notes:
            snapped_start = round(note.start_time / self.grid_size) * self.grid_size
            snapped_end = round(note.end_time / self.grid_size) * self.grid_size

            # ensure minimum duration of 1 grid unit
            if snapped_end <= snapped_start:
                snapped_end = snapped_start + self.grid_size
            
            quantized.append(Note(
                midi_pitch=note.midi_pitch,
                note_name=note.note_name,
                start_time=snapped_start,
                end_time=snapped_end,
                confidence=note.confidence
            ))
        return quantized
        
        