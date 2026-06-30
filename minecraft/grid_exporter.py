"""Export quantized notes as grid-native Minecraft song JSON."""

import json
import os
from collections import defaultdict

from models.note import Note
from minecraft.note_block_mapper import NoteBlockMapper
from minecraft.song_schema import GridNoteEvent, MinecraftSong, SongTrack
from transcription.quantizer import Quantizer


class GridExporter:
    """
    Build a MinecraftSong from transcribed notes.

    Uses Quantizer.to_grid_positions() so every event aligns to the
    redstone clock grid.
    """

    def __init__(self, default_instrument: str = "harp"):
        self.mapper = NoteBlockMapper(default_instrument=default_instrument)

    def from_notes(
        self,
        notes: list[Note],
        tempo_bpm: float,
        subdivisions: int = 4,
    ) -> MinecraftSong:
        if not notes:
            quantizer = Quantizer(tempo=tempo_bpm, subdivisions=subdivisions)
            return MinecraftSong(
                tempo_bpm=tempo_bpm,
                subdivisions=subdivisions,
                grid_size_seconds=quantizer.grid_size,
                ticks_per_beat=subdivisions,
                total_grid_length=0,
                tracks=[],
            )

        quantizer = Quantizer(tempo=tempo_bpm, subdivisions=subdivisions)
        grid_positions = quantizer.to_grid_positions(notes)

        tracks_by_key: dict[tuple[int, str], list[GridNoteEvent]] = defaultdict(list)
        track_instruments: dict[tuple[int, str], str] = {}
        max_grid = 0

        for entry in grid_positions:
            note: Note = entry["note"]
            mapping = self.mapper.map_note(note)
            event = GridNoteEvent(
                grid_start=entry["grid_start"],
                grid_duration=entry["grid_duration"],
                midi_pitch=mapping["midi_pitch"],
                note_name=mapping["note_name"],
                block_note=mapping["block_note"],
                block_instrument=mapping["block_instrument"],
                confidence=mapping["confidence"],
            )

            track_name = note.stem_name or "mix"
            key = (note.voice_id, track_name)
            tracks_by_key[key].append(event)
            track_instruments[key] = mapping["block_instrument"]
            max_grid = max(max_grid, entry["grid_end"])

        tracks: list[SongTrack] = []
        for (voice_id, name), events in sorted(tracks_by_key.items(), key=lambda x: x[0][0]):
            events.sort(key=lambda e: e.grid_start)
            tracks.append(SongTrack(
                name=name,
                voice_id=voice_id,
                instrument=track_instruments[(voice_id, name)],
                events=events,
            ))

        return MinecraftSong(
            tempo_bpm=tempo_bpm,
            subdivisions=subdivisions,
            grid_size_seconds=quantizer.grid_size,
            ticks_per_beat=subdivisions,
            total_grid_length=max_grid,
            tracks=tracks,
        )

    def write_json(self, song: MinecraftSong, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(song.to_dict(), f, indent=2)
        print(f"[GridExporter] Saved Minecraft song to: {output_path}")
        return output_path

    def export_notes(
        self,
        notes: list[Note],
        tempo_bpm: float,
        output_path: str,
        subdivisions: int = 4,
    ) -> str:
        song = self.from_notes(notes, tempo_bpm, subdivisions)
        return self.write_json(song, output_path)
