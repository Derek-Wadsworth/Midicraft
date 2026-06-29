import numpy as np
from models.note import Note


class NoteCleaner:
    """
    Filters and smooths raw note detections from NoteBuilder.

    The raw output of pitch detection is noisy:
      - Very short "notes" that are actually glitches or transients
      - Low-confidence detections that are likely wrong
      - Duplicate notes that are the same pitch played at the same time
      - Pitch wobble: a held note detected as many rapid micro-pitch changes

    This class removes or corrects those issues before export.
    Think of it as the "post-processing" step that separates
    a usable transcription from a garbage one.
    """

    def __init__(
        self,
        min_duration_s: float = 0.05,
        min_confidence: float = 0.4,
        merge_gap_s: float = 0.05,
    ):
        """
        Args:
            min_duration_s:  Discard notes shorter than this (likely artifacts).
                             50ms is about the shortest a human can perceive as a distinct note.
            min_confidence:  Discard notes below this confidence score.
                             0.4 is a reasonable threshold — below this, pYIN is guessing.
            merge_gap_s:     If two notes of the same pitch are separated by less than
                             this gap, merge them into one note (the gap was likely
                             a brief unvoiced frame, not a real note ending).
        """
        self.min_duration_s = min_duration_s
        self.min_confidence = min_confidence
        self.merge_gap_s = merge_gap_s

    def clean(self, notes: list[Note]) -> list[Note]:
        """
        Run all cleaning steps in order.

        Pipeline:
            1. Remove invalid notes (zero/negative duration, out-of-range pitch)
            2. Remove low-confidence detections
            3. Remove notes that are too short
            4. Merge same-pitch notes with tiny gaps between them
            5. Sort by start time

        Returns:
            Cleaned list of Note objects.
        """
        if not notes:
            return []

        notes = self._remove_invalid(notes)
        notes = self._remove_low_confidence(notes)
        notes = self._remove_short(notes)
        notes = self._merge_adjacent(notes)
        notes = sorted(notes, key=lambda n: n.start_time)

        return notes

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------

    def _remove_invalid(self, notes: list[Note]) -> list[Note]:
        """Remove notes with invalid pitch or non-positive duration."""
        valid = [n for n in notes if n.is_valid()]
        removed = len(notes) - len(valid)
        if removed > 0:
            print(f"  [Cleaner] Removed {removed} invalid notes")
        return valid

    def _remove_low_confidence(self, notes: list[Note]) -> list[Note]:
        """Remove notes below the confidence threshold."""
        filtered = [n for n in notes if n.confidence >= self.min_confidence]
        removed = len(notes) - len(filtered)
        if removed > 0:
            print(f"  [Cleaner] Removed {removed} low-confidence notes (< {self.min_confidence:.2f})")
        return filtered

    def _remove_short(self, notes: list[Note]) -> list[Note]:
        """Remove notes shorter than min_duration_s."""
        filtered = [n for n in notes if n.duration >= self.min_duration_s]
        removed = len(notes) - len(filtered)
        if removed > 0:
            print(f"  [Cleaner] Removed {removed} short notes (< {self.min_duration_s*1000:.0f}ms)")
        return filtered

    def _merge_adjacent(self, notes: list[Note]) -> list[Note]:
        """
        Merge consecutive notes of the same pitch if the gap between
        them is smaller than merge_gap_s.

        Example:
            Note(C4, 0.0-0.48) + gap(0.02s) + Note(C4, 0.50-0.98)
            → Note(C4, 0.0-0.98)

        Why this matters:
            Pitch detectors sometimes drop out for a frame in the middle
            of a held note. Without merging, you'd get two half-notes
            instead of one whole note. This sounds wrong in MIDI playback.
        """
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        merged = [sorted_notes[0]]

        for current in sorted_notes[1:]:
            prev = merged[-1]
            gap = current.start_time - prev.end_time
            same_pitch = current.midi_pitch == prev.midi_pitch

            if same_pitch and gap <= self.merge_gap_s:
                # extend the previous note to cover the current one
                merged[-1] = Note(
                    midi_pitch=prev.midi_pitch,
                    note_name=prev.note_name,
                    start_time=prev.start_time,
                    end_time=current.end_time,
                    confidence=max(prev.confidence, current.confidence),
                    frequency=prev.frequency,
                )
            else:
                merged.append(current)

        n_merged = len(notes) - len(merged)
        if n_merged > 0:
            print(f"  [Cleaner] Merged {n_merged} note fragments (gap < {self.merge_gap_s*1000:.0f}ms)")

        return merged

    # ------------------------------------------------------------------
    # Optional: pitch smoothing
    # ------------------------------------------------------------------

    def smooth_pitch_sequence(self, notes: list[Note], window: int = 3) -> list[Note]:
        """
        Apply a median filter to the pitch sequence to reduce rapid
        pitch oscillation (vibrato, pitch bend noise).

        This is optional — use it when your source has a lot of pitch wobble
        (e.g. voice, violin) but not when precision matters (piano, marimba).

        Args:
            notes:  Sorted list of notes.
            window: Median filter window size (odd number). Larger = smoother.

        Returns:
            Notes with smoothed midi_pitch values.
        """
        if len(notes) < window:
            return notes

        pitches = np.array([n.midi_pitch for n in notes])
        smoothed_pitches = np.array([
            int(np.median(pitches[max(0, i - window//2):i + window//2 + 1]))
            for i in range(len(pitches))
        ])

        result = []
        try:
            import librosa
            for note, new_pitch in zip(notes, smoothed_pitches):
                result.append(Note(
                    midi_pitch=int(new_pitch),
                    note_name=librosa.midi_to_note(int(new_pitch)),
                    start_time=note.start_time,
                    end_time=note.end_time,
                    confidence=note.confidence,
                    frequency=note.frequency,
                ))
        except ImportError:
            return notes  # skip smoothing if librosa not available

        return result
