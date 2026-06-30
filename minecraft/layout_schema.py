"""Block placement schema for Minecraft sequencer builds."""

from dataclasses import dataclass, field, asdict
from typing import Any


# Block-state keys valid in /setblock per block type
SETBLOCK_PROPERTIES: dict[str, frozenset[str]] = {
    "note_block": frozenset({"instrument", "note", "powered"}),
    "repeater": frozenset({"facing", "delay"}),
    "redstone_torch": frozenset({"facing"}),
    "redstone_wire": frozenset(),
}


@dataclass
class BlockPlacement:
    """A single block to place in the world."""

    x: int
    y: int
    z: int
    block: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "block": self.block,
            "properties": self.properties,
        }


@dataclass
class SequencerLayout:
    """
  3D block layout for a redstone note-block sequencer.

  Coordinate system (default):
    +X  = time (grid steps along the clock)
    +Z  = track lanes (one row per voice)
    Y=0 = redstone layer, Y=1 = note blocks
    """

    blocks_per_grid: int
    repeater_delay: int
    track_spacing: int
    clock_length: int
    track_count: int
    bounds: dict[str, int]
    blocks: list[BlockPlacement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "midicraft_layout_v1",
            "blocks_per_grid": self.blocks_per_grid,
            "repeater_delay": self.repeater_delay,
            "track_spacing": self.track_spacing,
            "clock_length": self.clock_length,
            "track_count": self.track_count,
            "bounds": self.bounds,
            "block_count": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def to_setblock_commands(self, namespace: str = "") -> list[str]:
        """Generate /setblock commands for in-game testing."""
        lines: list[str] = []
        for b in self.blocks:
            block_name = b.block.split(":")[-1]
            block_id = b.block if ":" in b.block else f"minecraft:{b.block}"
            allowed = SETBLOCK_PROPERTIES.get(block_name, frozenset())
            state_props = {k: v for k, v in b.properties.items() if k in allowed}
            if state_props:
                props = ",".join(f"{k}={v}" for k, v in state_props.items())
                block_id = f"{block_id}[{props}]"
            lines.append(f"setblock {b.x} {b.y} {b.z} {block_id}")
        return lines
