from dataclasses import dataclass
import numpy as np

@dataclass
class Note:
    midi_pitch: int #0-127
    note_name: str #e.g. 'A4'
    start_time: float #seconds
    end_time: float #seconds
    confidence: float #0.0 - 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class NoteBuilder:
    def build(
        self,
        f0: np.ndarray, # pitch per frame
        voiced_flag: np.ndarray, # is a note playing per frame
        times: np.ndarray, # timestamp per frame
    ) -> list[Note]:
        notes = []
        in_note = False
        current_pitches = []
        note_start = 0.0

        for i, (freq, voiced, time) in enumerate(zip(f0, voiced_flag, times)):
            if voiced and not in_note:
                # note starts
                in_note = True
                note_start = t
                current_pitches = [freq]

            elif voiced and in_note:
                # note continues
                current_pitches.append(freq)

            elif not voiced and in_note:
                # note ends
                in_note = False
                avg_freq = np.median(current_pitches) # use median for robustness
                midi = int(librosa.hz_to_midi(avg_freq))
                notes.append(Note(
                    midi_pitch=midi,
                    note_name=librosa.midi_to_note(midi),
                    start_time=note_start,
                    end_time=t,
                    confidence=float(np.mean(voiced_flag[max(0,i-len(current_pitches)):i]))
                ))
                current_pitches = []
        return notes