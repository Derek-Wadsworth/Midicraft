"""
Generate 3D block placements from a grid-native MinecraftSong.

Layout (top-down, default):

    Z=0   [repeater][wire][repeater][wire]...   <- shared clock
    Z=2   [note]              [note]            <- track 0 (vocals)
    Z=4   [note]     [note]                     <- track 1 (bass)

    X increases with grid time (each step = blocks_per_grid blocks).
"""

import json
import os

from minecraft.layout_schema import BlockPlacement, SequencerLayout
from minecraft.song_schema import GridNoteEvent, MinecraftSong, SongTrack


# Minecraft instrument id → block state instrument name (1.20+)
INSTRUMENT_BLOCK_STATE: dict[str, str] = {
    "harp": "harp",
    "bass": "bass",
    "basedrum": "basedrum",
    "snare": "snare",
    "hat": "hat",
    "guitar": "guitar",
    "flute": "flute",
    "bell": "bell",
    "chime": "chime",
    "xylophone": "xylophone",
    "iron_xylophone": "iron_xylophone",
    "cow_bell": "cow_bell",
    "didgeridoo": "didgeridoo",
    "bit": "bit",
    "banjo": "banjo",
    "pling": "pling",
}


class SequencerLayoutGenerator:
    """Convert MinecraftSong → SequencerLayout block coordinates."""

    def __init__(
        self,
        blocks_per_grid: int = 2,
        track_spacing: int = 2,
        repeater_delay: int | None = None,
        clock_z: int = 0,
        note_y: int = 1,
        redstone_y: int = 0,
    ):
        self.blocks_per_grid = max(1, blocks_per_grid)
        self.track_spacing = max(1, track_spacing)
        self.repeater_delay = repeater_delay
        self.clock_z = clock_z
        self.note_y = note_y
        self.redstone_y = redstone_y

    @staticmethod
    def repeater_delay_for_grid(grid_size_seconds: float) -> int:
        """
        Pick repeater delay (1–4) from grid step duration.

        One redstone tick ≈ 0.1 seconds.
        """
        ticks = grid_size_seconds / 0.1
        delay = max(1, min(4, round(ticks)))
        return delay

    def from_song(self, song: MinecraftSong) -> SequencerLayout:
        delay = self.repeater_delay or self.repeater_delay_for_grid(song.grid_size_seconds)
        clock_length = max(song.total_grid_length, 1)
        blocks: list[BlockPlacement] = []

        # --- shared repeater clock along +X at z=clock_z ---
        for step in range(clock_length):
            x = step * self.blocks_per_grid
            blocks.append(BlockPlacement(
                x=x,
                y=self.redstone_y,
                z=self.clock_z,
                block="repeater",
                properties={"facing": "east", "delay": delay},
            ))
            wire_x = x + 1
            if wire_x < clock_length * self.blocks_per_grid:
                blocks.append(BlockPlacement(
                    x=wire_x,
                    y=self.redstone_y,
                    z=self.clock_z,
                    block="redstone_wire",
                    properties={},
                ))

        # Clock origin comparator output (optional start pulse)
        blocks.append(BlockPlacement(
            x=-1,
            y=self.redstone_y,
            z=self.clock_z,
            block="redstone_torch",
            properties={"facing": "east"},
        ))

        max_x = 0
        max_z = self.clock_z

        # --- note block rows per track ---
        for track_idx, track in enumerate(song.tracks):
            z = self.clock_z + (track_idx + 1) * self.track_spacing
            max_z = max(max_z, z)

            for event in track.events:
                x = event.grid_start * self.blocks_per_grid
                max_x = max(max_x, x)

                instrument = INSTRUMENT_BLOCK_STATE.get(
                    event.block_instrument, event.block_instrument
                )
                blocks.append(BlockPlacement(
                    x=x,
                    y=self.note_y,
                    z=z,
                    block="note_block",
                    properties={
                        "instrument": instrument,
                        "note": event.block_note,
                        "powered": "false",
                        "grid_duration": event.grid_duration,
                        "note_name": event.note_name,
                        "track": track.name,
                    },
                ))

                # Trigger wire on redstone layer (mod or player connects to clock)
                blocks.append(BlockPlacement(
                    x=x,
                    y=self.redstone_y,
                    z=z,
                    block="redstone_wire",
                    properties={"track": track.name},
                ))

                # Sustained notes: extra wire segments for grid_duration > 1
                for offset in range(1, event.grid_duration):
                    dx = x + offset * self.blocks_per_grid
                    if dx <= clock_length * self.blocks_per_grid:
                        max_x = max(max_x, dx)
                        blocks.append(BlockPlacement(
                            x=dx,
                            y=self.redstone_y,
                            z=z,
                            block="redstone_wire",
                            properties={"sustain": True, "track": track.name},
                        ))

        bounds = {
            "min_x": -1,
            "max_x": max_x + 1,
            "min_y": self.redstone_y,
            "max_y": self.note_y,
            "min_z": self.clock_z,
            "max_z": max_z,
        }

        return SequencerLayout(
            blocks_per_grid=self.blocks_per_grid,
            repeater_delay=delay,
            track_spacing=self.track_spacing,
            clock_length=clock_length,
            track_count=len(song.tracks),
            bounds=bounds,
            blocks=blocks,
        )

    @staticmethod
    def song_from_dict(data: dict) -> MinecraftSong:
        """Load MinecraftSong from *_song.json dict."""
        tracks = []
        for t in data.get("tracks", []):
            events = [
                GridNoteEvent(
                    grid_start=e["grid_start"],
                    grid_duration=e["grid_duration"],
                    midi_pitch=e["midi_pitch"],
                    note_name=e["note_name"],
                    block_note=e["block_note"],
                    block_instrument=e["block_instrument"],
                    confidence=e.get("confidence", 1.0),
                )
                for e in t.get("events", [])
            ]
            tracks.append(SongTrack(
                name=t["name"],
                voice_id=t["voice_id"],
                instrument=t["instrument"],
                events=events,
            ))

        return MinecraftSong(
            tempo_bpm=data["tempo_bpm"],
            subdivisions=data["subdivisions"],
            grid_size_seconds=data["grid_size_seconds"],
            ticks_per_beat=data["ticks_per_beat"],
            total_grid_length=data["total_grid_length"],
            tracks=tracks,
        )

    @staticmethod
    def load_song_json(path: str) -> MinecraftSong:
        with open(path, encoding="utf-8") as f:
            return SequencerLayoutGenerator.song_from_dict(json.load(f))

    def write_layout_json(self, layout: SequencerLayout, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(layout.to_dict(), f, indent=2)
        print(f"[SequencerLayout] Saved layout to: {output_path}")
        return output_path

    def write_mcfunction(self, layout: SequencerLayout, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        lines = ["# Midicraft sequencer — generated setblock commands", ""]
        lines.extend(layout.to_setblock_commands())
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[SequencerLayout] Saved mcfunction to: {output_path}")
        return output_path

    def export_from_song(self, song: MinecraftSong, layout_path: str, mcfunction_path: str | None = None) -> SequencerLayout:
        layout = self.from_song(song)
        self.write_layout_json(layout, layout_path)
        if mcfunction_path:
            self.write_mcfunction(layout, mcfunction_path)
        return layout
