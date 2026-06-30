import numpy as np
from models.note import Note

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class NoteBuilder:
    """
    Converts raw pitch detection output into a list of Note objects.

    Input:  Three parallel arrays from PitchDetector —
              f0[]          — frequency at each frame (0 = unvoiced)
              voiced_flag[] — True where a note is playing
              times[]       — timestamp (seconds) of each frame

    Output: List of Note objects with start_time, end_time, pitch, confidence.

    The core logic is a simple state machine:
        State: IN_NOTE or NOT_IN_NOTE
        Transitions:
            NOT_IN_NOTE + voiced frame  → IN_NOTE (record start time, collect pitches)
            IN_NOTE     + voiced frame  → stay IN_NOTE (accumulate pitch samples)
            IN_NOTE     + unvoiced frame → NOT_IN_NOTE (finalize note, compute avg pitch)
    """

    def build(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        times: np.ndarray,
    ) -> list[Note]:
        """
        Build a list of Note objects from pitch detector output.

        Args:
            f0:          Frequency per frame in Hz. Shape: (n_frames,)
            voiced_flag: Boolean voiced/unvoiced per frame. Shape: (n_frames,)
            times:       Timestamp (seconds) per frame. Shape: (n_frames,)

        Returns:
            List of Note objects, roughly sorted by start_time.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for MIDI conversion.")

        notes = []
        in_note = False
        note_start = 0.0
        pitch_samples = []      # raw Hz values while note is active
        confidence_samples = [] # voiced_flag is bool, so we use it as 0/1 confidence

        for freq, voiced, t in zip(f0, voiced_flag, times):
            if voiced and not in_note:
                # ---- NOTE START ----
                in_note = True
                note_start = float(t)
                pitch_samples = [float(freq)]
                confidence_samples = [1.0]

            elif voiced and in_note:
                # ---- NOTE CONTINUES ----
                pitch_samples.append(float(freq))
                confidence_samples.append(1.0)

            elif not voiced and in_note:
                # ---- NOTE END ----
                in_note = False
                note = self._finalize_note(note_start, float(t), pitch_samples, confidence_samples)
                if note:
                    notes.append(note)
                pitch_samples = []
                confidence_samples = []

        # handle note still active at end of signal
        if in_note and pitch_samples:
            note = self._finalize_note(note_start, float(times[-1]), pitch_samples, confidence_samples)
            if note:
                notes.append(note)

        return notes

    def build_with_confidence(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        voiced_prob: np.ndarray,
        times: np.ndarray,
    ) -> list[Note]:
        """
        Like build() but uses voiced_prob for per-note confidence scores.
        Use this with PitchDetector.detect_with_confidence().
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required.")

        notes = []
        in_note = False
        note_start = 0.0
        pitch_samples = []
        prob_samples = []

        for freq, voiced, prob, t in zip(f0, voiced_flag, voiced_prob, times):
            if voiced and not in_note:
                in_note = True
                note_start = float(t)
                pitch_samples = [float(freq)]
                prob_samples = [float(prob)]

            elif voiced and in_note:
                pitch_samples.append(float(freq))
                prob_samples.append(float(prob))

            elif not voiced and in_note:
                in_note = False
                note = self._finalize_note(note_start, float(t), pitch_samples, prob_samples)
                if note:
                    notes.append(note)
                pitch_samples = []
                prob_samples = []

        if in_note and pitch_samples:
            note = self._finalize_note(note_start, float(times[-1]), pitch_samples, prob_samples)
            if note:
                notes.append(note)

        return notes

    def _finalize_note(
        self,
        start: float,
        end: float,
        pitch_samples: list[float],
        confidence_samples: list[float],
    ) -> Note | None:
        """
        Convert accumulated pitch samples into a single Note.

        Pitch estimation:
            We use the MEDIAN of all frequency samples, not the mean.
            Median is more robust to pitch tracking glitches — a few bad
            frames won't drag the pitch estimate far off.
        """
        if not pitch_samples or end <= start:
            return None

        # median frequency → most representative pitch of the note
        avg_freq = float(np.median(pitch_samples))
        if avg_freq <= 0:
            return None

        midi_pitch = int(round(librosa.hz_to_midi(avg_freq)))
        midi_pitch = max(0, min(127, midi_pitch))  # clamp to valid MIDI range

        confidence = float(np.mean(confidence_samples))

        return Note(
            midi_pitch=midi_pitch,
            note_name=librosa.midi_to_note(midi_pitch, unicode=False),
            start_time=start,
            end_time=end,
            confidence=confidence,
            frequency=avg_freq,
        )
