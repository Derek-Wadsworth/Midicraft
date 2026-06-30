# Midicraft

Audio transcription pipeline that detects pitch, quantizes to a musical grid, and generates MIDI/JSON for Minecraft redstone sequencers.

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| **mono** (default) | `python main.py song.mp3` | Single-voice pYIN transcription |
| **stems** | `python main.py song.mp3 --mode stems` | Demucs separation + per-stem mono transcription |
| **poly** | `python main.py song.mp3 --mode poly` | Basic Pitch polyphonic transcription |
| **stems+poly** | `python main.py song.mp3 --mode stems+poly` | Stems + polyphonic on `other` |

## Install

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-stems.txt   # Demucs stem separation
pip install -r requirements-poly.txt    # Basic Pitch polyphonic mode
```

## Examples

```bash
# Monophonic (existing behavior)
python main.py song.mp3 --no-plot

# Separate stems only (no transcription)
python main.py song.mp3 --separate-only

# Full stem pipeline → multi-track MIDI
python main.py song.mp3 --mode stems --export-formats midi json --no-plot

# Polyphonic on piano/guitar-heavy audio
python main.py song.mp3 --mode poly

# Stems with poly on the "other" stem
python main.py song.mp3 --mode stems+poly --stem-modes other:poly

# Re-use pre-separated stems
python main.py song.mp3 --mode stems --stems-dir ./output/song_stems
```

## Output

- **mono**: `./output/song.mid`, `song_notes.txt`
- **stems**: `./output/song_multitrack.mid`, `song_stems/vocals.wav`, etc.
- **poly**: `./output/song_poly.mid`

JSON exports include `voice_id` and `stem_name` per note for Minecraft lane mapping.
