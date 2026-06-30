import os
import csv
import json
from models.note import Note
from models.transcription_result import MergedTranscriptionResult


class TextExporter:
    """
    Exports transcribed notes to human-readable text, CSV, and JSON formats.

    Useful for:
      - Debugging your pipeline output
      - Sharing results without needing a MIDI player
      - Feeding into the Minecraft note block converter later
    """

    # ------------------------------------------------------------------
    # Plain text
    # ------------------------------------------------------------------

    def to_text(self, notes: list[Note], output_path: str = None) -> str:
        """
        Export notes as a formatted plain text table.

        Example output:
            #   Note   MIDI   Start      End      Duration   Confidence
            1   C4     60     0.000s     0.480s   0.480s     0.95
            2   E4     64     0.500s     0.980s   0.480s     0.91
        """
        lines = []
        lines.append("=" * 65)
        lines.append("TRANSCRIPTION RESULTS")
        lines.append("=" * 65)
        has_voice = any(n.voice_id != 0 or n.stem_name for n in notes)
        if has_voice:
            lines.append(f"{'#':<5} {'Stem':<8} {'Ch':>2}  {'Note':<6} {'MIDI':>4}  {'Start':>8}  {'End':>8}  {'Duration':>9}  {'Conf':>5}")
            lines.append("-" * 80)
            for i, note in enumerate(notes, 1):
                stem = note.stem_name or "-"
                lines.append(
                    f"{i:<5} {stem:<8} {note.voice_id:>2}  {note.note_name:<6} {note.midi_pitch:>4}  "
                    f"{note.start_time:>7.3f}s  {note.end_time:>7.3f}s  "
                    f"{note.duration:>8.3f}s  {note.confidence:>5.2f}"
                )
            lines.append("-" * 80)
        else:
            lines.append(f"{'#':<5} {'Note':<6} {'MIDI':>4}  {'Start':>8}  {'End':>8}  {'Duration':>9}  {'Conf':>5}")
            lines.append("-" * 65)
            for i, note in enumerate(notes, 1):
                lines.append(
                    f"{i:<5} {note.note_name:<6} {note.midi_pitch:>4}  "
                    f"{note.start_time:>7.3f}s  {note.end_time:>7.3f}s  "
                    f"{note.duration:>8.3f}s  {note.confidence:>5.2f}"
                )
            lines.append("-" * 65)
        lines.append(f"Total notes: {len(notes)}")
        if notes:
            lines.append(f"Total duration: {max(n.end_time for n in notes):.3f}s")
            avg_conf = sum(n.confidence for n in notes) / len(notes)
            lines.append(f"Average confidence: {avg_conf:.2f}")
        lines.append("=" * 65)

        result = "\n".join(lines)

        if output_path:
            self._write(result, output_path)

        return result

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, notes: list[Note], output_path: str) -> str:
        """
        Export notes as a CSV file.

        Columns: index, note_name, midi_pitch, start_time, end_time, duration, confidence, frequency_hz

        Useful for loading into spreadsheets or pandas for further analysis.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index', 'stem_name', 'voice_id', 'note_name', 'midi_pitch',
                'start_time', 'end_time', 'duration', 'confidence', 'frequency_hz',
            ])
            for i, note in enumerate(notes, 1):
                writer.writerow([
                    i,
                    note.stem_name or '',
                    note.voice_id,
                    note.note_name,
                    note.midi_pitch,
                    round(note.start_time, 4),
                    round(note.end_time, 4),
                    round(note.duration, 4),
                    round(note.confidence, 4),
                    round(note.frequency, 2),
                ])

        print(f"[TextExporter] Saved CSV to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, notes: list[Note], output_path: str = None) -> str:
        """
        Export notes as JSON.

        This is the format we'll use later to pass data into the
        Minecraft note block generator — each note maps to a block placement.
        """
        data = {
            "total_notes": len(notes),
            "total_duration": round(max((n.end_time for n in notes), default=0), 4),
            "notes": [
                {
                    "index": i,
                    "stem_name": note.stem_name,
                    "voice_id": note.voice_id,
                    "note_name": note.note_name,
                    "midi_pitch": note.midi_pitch,
                    "start_time": round(note.start_time, 4),
                    "end_time": round(note.end_time, 4),
                    "duration": round(note.duration, 4),
                    "confidence": round(note.confidence, 4),
                    "frequency_hz": round(note.frequency, 2),
                }
                for i, note in enumerate(notes, 1)
            ]
        }

        result = json.dumps(data, indent=2)

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"[TextExporter] Saved JSON to: {output_path}")

        return result

    def to_json_multitrack(self, merged: MergedTranscriptionResult, output_path: str) -> str:
        """Export multi-stem transcription with per-track note lists."""
        tracks = []
        for stem_result in merged.stem_results:
            tracks.append({
                "name": stem_result.source_label,
                "voice_id": stem_result.notes[0].voice_id if stem_result.notes else 0,
                "note_count": len(stem_result.notes),
                "notes": [
                    {
                        "index": i,
                        "stem_name": note.stem_name,
                        "voice_id": note.voice_id,
                        "note_name": note.note_name,
                        "midi_pitch": note.midi_pitch,
                        "start_time": round(note.start_time, 4),
                        "end_time": round(note.end_time, 4),
                        "duration": round(note.duration, 4),
                        "confidence": round(note.confidence, 4),
                        "frequency_hz": round(note.frequency, 2),
                    }
                    for i, note in enumerate(stem_result.notes, 1)
                ],
            })

        data = {
            "tempo_bpm": round(merged.tempo, 2),
            "total_notes": len(merged.notes),
            "total_duration": round(max((n.end_time for n in merged.notes), default=0), 4),
            "tracks": tracks,
            "notes": [
                {
                    "index": i,
                    "stem_name": note.stem_name,
                    "voice_id": note.voice_id,
                    "note_name": note.note_name,
                    "midi_pitch": note.midi_pitch,
                    "start_time": round(note.start_time, 4),
                    "end_time": round(note.end_time, 4),
                    "duration": round(note.duration, 4),
                    "confidence": round(note.confidence, 4),
                    "frequency_hz": round(note.frequency, 2),
                }
                for i, note in enumerate(merged.notes, 1)
            ],
        }

        result = json.dumps(data, indent=2)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"[TextExporter] Saved multi-track JSON to: {output_path}")
        return result

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self, notes: list[Note]) -> dict:
        """
        Return a dictionary of summary statistics about the transcription.
        Useful for quick sanity checks on your pipeline output.
        """
        if not notes:
            return {"error": "No notes detected"}

        pitches = [n.midi_pitch for n in notes]
        durations = [n.duration for n in notes]
        confidences = [n.confidence for n in notes]

        import librosa
        return {
            "total_notes": len(notes),
            "total_duration_s": round(max(n.end_time for n in notes), 3),
            "pitch_range": {
                "lowest": librosa.midi_to_note(min(pitches)),
                "highest": librosa.midi_to_note(max(pitches)),
                "span_semitones": max(pitches) - min(pitches),
            },
            "duration": {
                "shortest_ms": round(min(durations) * 1000, 1),
                "longest_ms": round(max(durations) * 1000, 1),
                "average_ms": round(sum(durations) / len(durations) * 1000, 1),
            },
            "confidence": {
                "average": round(sum(confidences) / len(confidences), 3),
                "min": round(min(confidences), 3),
                "max": round(max(confidences), 3),
                "low_confidence_notes": sum(1 for c in confidences if c < 0.5),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, content: str, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[TextExporter] Saved text to: {path}")