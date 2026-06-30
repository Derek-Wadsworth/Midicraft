import os
from models.note import Note

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


class MidiWriter:
    """
    Converts a list of Note objects into a standard MIDI file (.mid).

    MIDI basics you need to know:
      - MIDI time is measured in "ticks", not seconds.
      - Ticks per beat (ticks_per_beat) defines the resolution.
      - A "tempo" MetaMessage sets how many microseconds per beat.
      - note_on  velocity > 0  → note starts
      - note_off velocity = 0  → note ends
      - All times in a MIDI track are *delta times* (time since last event),
        not absolute times.
    """

    def __init__(self, ticks_per_beat: int = 480):
        """
        Args:
            ticks_per_beat: Resolution of the MIDI file. 480 is standard.
                            Higher = more precise timing.
        """
        self.ticks_per_beat = ticks_per_beat

    def _seconds_to_ticks(self, seconds: float, tempo: int) -> int:
        """
        Convert seconds to MIDI ticks.

        Formula:
            ticks = seconds * (ticks_per_beat / seconds_per_beat)
            seconds_per_beat = tempo / 1_000_000  (tempo is in microseconds)
        """
        seconds_per_beat = tempo / 1_000_000
        ticks_per_second = self.ticks_per_beat / seconds_per_beat
        return int(seconds * ticks_per_second)

    def write(
        self,
        notes: list[Note],
        output_path: str,
        tempo_bpm: float = 120.0,
        channel: int = 0,
        velocity: int = 80,
    ) -> str:
        """
        Write a list of Notes to a MIDI file.

        Args:
            notes:       List of Note objects to write.
            output_path: Where to save the .mid file.
            tempo_bpm:   Beats per minute for the MIDI file.
            channel:     MIDI channel (0-15). Channel 9 is drums.
            velocity:    How hard notes are struck (0-127). 80 is medium.

        Returns:
            The path of the written file.
        """
        if not MIDO_AVAILABLE:
            print("[MidiWriter] 'mido' not installed. Run: pip install mido")
            print("[MidiWriter] Falling back to text export...")
            return self._write_fallback(notes, output_path)

        if not notes:
            print("[MidiWriter] No notes to write.")
            return output_path

        # convert BPM to microseconds per beat (MIDI tempo format)
        tempo_us = int(60_000_000 / tempo_bpm)

        midi = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        midi.tracks.append(track)

        # write tempo at the start of the track
        track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # build a flat list of events: (absolute_tick, type, pitch)
        events = []
        for note in notes:
            if not note.is_valid():
                continue
            on_tick  = self._seconds_to_ticks(note.start_time, tempo_us)
            off_tick = self._seconds_to_ticks(note.end_time,   tempo_us)
            events.append((on_tick,  'note_on',  note.midi_pitch))
            events.append((off_tick, 'note_off', note.midi_pitch))

        # sort by absolute tick time
        events.sort(key=lambda e: e[0])

        # convert absolute ticks → delta ticks and append to track
        last_tick = 0
        for abs_tick, event_type, pitch in events:
            delta = abs_tick - last_tick
            last_tick = abs_tick
            if event_type == 'note_on':
                track.append(Message('note_on',  channel=channel, note=pitch, velocity=velocity, time=delta))
            else:
                track.append(Message('note_off', channel=channel, note=pitch, velocity=0,        time=delta))

        # ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        midi.save(output_path)
        print(f"[MidiWriter] Saved MIDI to: {output_path}")
        return output_path

    def _write_fallback(self, notes: list[Note], output_path: str) -> str:
        """Write a simple text representation when mido is not available."""
        txt_path = output_path.replace('.mid', '_notes.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# Transcribed Notes (mido not available)\n")
            f.write(f"# {'Note':<6} {'MIDI':>4} {'Start':>8} {'End':>8} {'Duration':>10}\n")
            for note in notes:
                f.write(f"  {note.note_name:<6} {note.midi_pitch:>4} {note.start_time:>8.3f} {note.end_time:>8.3f} {note.duration:>10.3f}\n")
        print(f"[MidiWriter] Saved note list to: {txt_path}")
        return txt_path
