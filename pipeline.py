"""
pipeline.py — Orchestrates transcription pipelines (mono, stems, poly).

Usage:
    from pipeline import MonophonicPipeline, StemPipeline, PolyphonicPipeline

    notes = MonophonicPipeline("song.mp3").run()
    result = StemPipeline("song.mp3").run()
"""

import os

import librosa
import numpy as np

from audio.loader import AudioLoader
from audio.preprocessor import Preprocessor
from features.onset import OnsetDetector
from features.pitch import PitchDetector
from features.polyphonic import PolyphonicDetector
from models.note import Note
from models.transcription_result import MergedTranscriptionResult, TranscriptionResult
from output.midi_writer import MidiWriter
from output.text_export import TextExporter
from output.visualizer import Visualizer
from separation.stem_config import (
    ALL_STEMS,
    DEFAULT_STEMS,
    STEM_CHANNEL_MAP,
    STEM_PROFILES,
)
from separation.stem_separator import StemSeparator
from transcription.cleaner import NoteCleaner
from transcription.note_builder import NoteBuilder
from transcription.poly_note_builder import PolyNoteBuilder
from transcription.quantizer import Quantizer


def export_minecraft_song(
    notes: list,
    tempo_bpm: float,
    output_path: str,
    subdivisions: int = 4,
) -> str:
    """Write grid-native Minecraft song JSON."""
    from minecraft.grid_exporter import GridExporter
    return GridExporter().export_notes(notes, tempo_bpm, output_path, subdivisions)


class MonophonicPipeline:
    """
    End-to-end monophonic melody transcription on a single waveform.

    Stages:
        1. Load / preprocess
        2. Pitch (pYIN)
        3. Tempo
        4. Build notes
        5. Clean
        6. Quantize
        7. Export
    """

    def __init__(
        self,
        filepath: str | None = None,
        sample_rate: int = 22050,
        min_confidence: float = 0.4,
        min_note_duration_s: float = 0.05,
        quantize_subdivisions: int = 4,
        output_dir: str = "./output",
        fmin: str = "C2",
        fmax: str = "C7",
    ):
        self.filepath = filepath
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.min_confidence = min_confidence
        self.min_note_duration_s = min_note_duration_s
        self.quantize_subdivisions = quantize_subdivisions
        self.fmin = fmin
        self.fmax = fmax

        self.loader = AudioLoader(target_sr=sample_rate)
        self.preprocessor = Preprocessor()
        self.onset = OnsetDetector(sr=sample_rate)
        self.builder = NoteBuilder()
        self.cleaner = NoteCleaner(
            min_confidence=min_confidence,
            min_duration_s=min_note_duration_s,
        )
        self.midi_writer = MidiWriter()
        self.visualizer = Visualizer(sr=sample_rate)
        self.exporter = TextExporter()

        self.y: np.ndarray | None = None
        self.sr: int = sample_rate
        self.f0: np.ndarray | None = None
        self.voiced_flag: np.ndarray | None = None
        self.voiced_prob: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.tempo: float = 120.0
        self.notes: list[Note] = []

    def transcribe_signal(
        self,
        y: np.ndarray,
        sr: int,
        label: str = "mix",
        tempo: float | None = None,
        fmin: str | None = None,
        fmax: str | None = None,
        min_confidence: float | None = None,
        voice_id: int = 0,
        stem_name: str | None = None,
        skip_preprocess: bool = False,
    ) -> TranscriptionResult:
        """Run transcription on an in-memory mono waveform."""
        if not skip_preprocess:
            y = self.preprocessor.normalize(y)
            y = self.preprocessor.trim_silence(y, sr)
            y = self.preprocessor.apply_preemphasis(y)

        pitch = PitchDetector(sr=sr, fmin=fmin or self.fmin, fmax=fmax or self.fmax)
        f0, voiced_flag, voiced_prob = pitch.detect_with_confidence(y)
        times = librosa.times_like(f0, sr=sr)

        if tempo is None:
            tempo = self.onset.detect_tempo(y)

        notes = self.builder.build_with_confidence(f0, voiced_flag, voiced_prob, times)

        cleaner = NoteCleaner(
            min_confidence=min_confidence if min_confidence is not None else self.min_confidence,
            min_duration_s=self.min_note_duration_s,
        )
        notes = cleaner.clean(notes)

        quantizer = Quantizer(tempo=tempo, subdivisions=self.quantize_subdivisions)
        notes = quantizer.snap_to_grid(notes)

        for note in notes:
            note.voice_id = voice_id
            if stem_name:
                note.stem_name = stem_name

        return TranscriptionResult(
            notes=notes,
            tempo=tempo,
            source_label=label,
            f0=f0,
            voiced_flag=voiced_flag,
            voiced_prob=voiced_prob,
            times=times,
            y=y,
            sr=sr,
        )

    def transcribe_polyphonic(
        self,
        y: np.ndarray,
        sr: int,
        label: str = "poly",
        tempo: float | None = None,
        voice_id: int = 0,
        stem_name: str | None = None,
        min_confidence: float | None = None,
    ) -> TranscriptionResult:
        """Run Basic Pitch polyphonic detection on a waveform."""
        builder = PolyNoteBuilder(min_confidence=min_confidence or self.min_confidence)
        notes = builder.build_from_array(y, sr, label=label)

        cleaner = NoteCleaner(
            min_confidence=min_confidence or self.min_confidence,
            min_duration_s=self.min_note_duration_s,
        )
        notes = cleaner.clean(notes)

        if tempo is None:
            tempo = self.onset.detect_tempo(y)

        quantizer = Quantizer(tempo=tempo, subdivisions=self.quantize_subdivisions)
        notes = quantizer.snap_to_grid(notes)

        for note in notes:
            note.voice_id = voice_id
            if stem_name:
                note.stem_name = stem_name

        return TranscriptionResult(
            notes=notes,
            tempo=tempo,
            source_label=label,
            y=y,
            sr=sr,
        )

    def run(
        self,
        export_midi: bool = True,
        export_text: bool = True,
        export_csv: bool = False,
        export_json: bool = False,
        export_song: bool = False,
        show_plot: bool = True,
        save_plot: bool = False,
    ) -> list[Note]:
        if not self.filepath:
            raise ValueError("filepath is required for MonophonicPipeline.run()")

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\n[1/7] Loading: {self.filepath}")
        self.y, self.sr = self.loader.load(self.filepath)
        print(f"      {len(self.y)/self.sr:.2f}s  |  {self.sr} Hz  |  {len(self.y):,} samples")

        print("[2/7] Preprocessing...")
        print("[3/7] Detecting pitch (pYIN)...")
        result = self.transcribe_signal(self.y, self.sr, label="mix", skip_preprocess=False)
        if result.voiced_flag is not None:
            pct = 100 * result.voiced_flag.sum() / len(result.voiced_flag)
            print(f"      {result.voiced_flag.sum()}/{len(result.voiced_flag)} frames voiced ({pct:.1f}%)")
        print(f"[4/7] Tempo: {result.tempo:.1f} BPM")
        print(f"[5/7] Building notes: {len(result.notes)} after clean+quantize")

        self._apply_result(result)

        print("[6/7] Quantized to grid")
        print("[7/7] Exporting...")
        self._step_export(export_midi, export_text, export_csv, export_json, export_song)

        if show_plot or save_plot:
            save_path = None
            if save_plot:
                base = os.path.splitext(os.path.basename(self.filepath))[0]
                save_path = os.path.join(self.output_dir, f"{base}_dashboard.png")
            self.visualizer.full_dashboard(
                self.y, self.f0, self.voiced_flag, self.notes, save_path=save_path
            )

        return self.notes

    def _apply_result(self, result: TranscriptionResult):
        self.notes = result.notes
        self.tempo = result.tempo
        self.f0 = result.f0
        self.voiced_flag = result.voiced_flag
        self.voiced_prob = result.voiced_prob
        self.times = result.times
        if result.y is not None:
            self.y = result.y
        self.sr = result.sr

    def _step_export(self, midi: bool, text: bool, csv: bool, json_export: bool, song: bool = False):
        base = os.path.splitext(os.path.basename(self.filepath))[0]

        if midi:
            path = os.path.join(self.output_dir, f"{base}.mid")
            self.midi_writer.write(self.notes, path, tempo_bpm=self.tempo)

        if text:
            path = os.path.join(self.output_dir, f"{base}_notes.txt")
            content = self.exporter.to_text(self.notes)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[TextExporter] Saved text to: {path}")

        if csv:
            path = os.path.join(self.output_dir, f"{base}_notes.csv")
            self.exporter.to_csv(self.notes, path)

        if json_export:
            path = os.path.join(self.output_dir, f"{base}_notes.json")
            self.exporter.to_json(self.notes, path)

        if song:
            path = os.path.join(self.output_dir, f"{base}_song.json")
            export_minecraft_song(
                self.notes, self.tempo, path, self.quantize_subdivisions
            )

    def print_notes(self):
        print(self.exporter.to_text(self.notes))

    def summary(self) -> dict:
        return self.exporter.summary(self.notes)


class StemPipeline:
    """Separate stems, transcribe each, merge into multi-voice output."""

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 22050,
        min_confidence: float = 0.4,
        min_note_duration_s: float = 0.05,
        quantize_subdivisions: int = 4,
        output_dir: str = "./output",
        stems: list[str] | None = None,
        stem_model: str = "htdemucs",
        device: str | None = None,
        stem_modes: dict[str, str] | None = None,
        stems_dir: str | None = None,
        separate_only: bool = False,
    ):
        self.filepath = filepath
        self.output_dir = output_dir
        self.stems = list(stems or DEFAULT_STEMS)
        self.stem_model = stem_model
        self.stems_dir = stems_dir
        self.separate_only = separate_only
        self.stem_modes = stem_modes or {s: "mono" for s in self.stems}

        self.mono = MonophonicPipeline(
            filepath=filepath,
            sample_rate=sample_rate,
            min_confidence=min_confidence,
            min_note_duration_s=min_note_duration_s,
            quantize_subdivisions=quantize_subdivisions,
            output_dir=output_dir,
        )
        self.separator = StemSeparator(model_name=stem_model, device=device)
        self.midi_writer = MidiWriter()
        self.exporter = TextExporter()
        self.visualizer = Visualizer(sr=sample_rate)

        self.result: MergedTranscriptionResult | None = None

    def run(
        self,
        export_midi: bool = True,
        export_text: bool = True,
        export_csv: bool = False,
        export_json: bool = False,
        export_song: bool = False,
        show_plot: bool = False,
        save_plot: bool = False,
    ) -> MergedTranscriptionResult:
        os.makedirs(self.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.filepath))[0]
        stems_out = self.stems_dir or os.path.join(self.output_dir, f"{base}_stems")

        if self.stems_dir and os.path.isdir(self.stems_dir):
            print(f"[StemPipeline] Loading existing stems from {self.stems_dir}")
            separated = self.separator.load_existing_stems(self.stems_dir, stems=self.stems)
        else:
            print(f"[StemPipeline] Separating stems -> {stems_out}")
            paths = self.separator.separate_and_save(
                self.filepath, stems_out, stems=list(ALL_STEMS)
            )
            if self.separate_only:
                print("[StemPipeline] --separate-only: skipping transcription.")
                return MergedTranscriptionResult(notes=[], tempo=120.0)

            separated = {}
            for name in self.stems:
                if name not in paths:
                    continue
                y_stem, sr_stem = self.mono.loader.load(paths[name])
                separated[name] = (y_stem, sr_stem)

        # Tempo from full mix (more stable than sparse stems)
        print("[StemPipeline] Detecting tempo from full mix...")
        y_mix, sr_mix = self.mono.loader.load(self.filepath)
        tempo = self.mono.onset.detect_tempo(y_mix)
        print(f"      {tempo:.1f} BPM")

        stem_results: list[TranscriptionResult] = []
        all_notes: list[Note] = []

        for stem_name in self.stems:
            if stem_name == "drums":
                print(f"[StemPipeline] Skipping pitch transcription for drums.")
                continue

            if stem_name not in separated:
                print(f"[StemPipeline] Stem '{stem_name}' not available, skipping.")
                continue

            y_stem, sr_stem = separated[stem_name]
            if self.mono.sample_rate != sr_stem:
                y_stem = librosa.resample(y_stem, orig_sr=sr_stem, target_sr=self.mono.sample_rate)
                sr_stem = self.mono.sample_rate

            profile = STEM_PROFILES.get(stem_name, {})
            voice_id = STEM_CHANNEL_MAP.get(stem_name, 0)
            mode = self.stem_modes.get(stem_name, "mono")

            print(f"[StemPipeline] Transcribing stem '{stem_name}' ({mode})...")

            if mode == "poly":
                result = self.mono.transcribe_polyphonic(
                    y_stem, sr_stem,
                    label=stem_name,
                    tempo=tempo,
                    voice_id=voice_id,
                    stem_name=stem_name,
                    min_confidence=profile.get("min_confidence"),
                )
            else:
                result = self.mono.transcribe_signal(
                    y_stem, sr_stem,
                    label=stem_name,
                    tempo=tempo,
                    fmin=profile.get("fmin"),
                    fmax=profile.get("fmax"),
                    min_confidence=profile.get("min_confidence"),
                    voice_id=voice_id,
                    stem_name=stem_name,
                )

            print(f"      {len(result.notes)} notes from {stem_name}")
            stem_results.append(result)
            all_notes.extend(result.notes)

        merged = MergedTranscriptionResult(
            notes=sorted(all_notes, key=lambda n: (n.start_time, n.voice_id)),
            tempo=tempo,
            stem_results=stem_results,
            y=y_mix,
            sr=sr_mix,
        )
        self.result = merged

        if export_midi:
            path = os.path.join(self.output_dir, f"{base}_multitrack.mid")
            self.midi_writer.write_multi_track(merged.notes, path, tempo_bpm=tempo)

        if export_text:
            path = os.path.join(self.output_dir, f"{base}_multitrack_notes.txt")
            content = self.exporter.to_text(merged.notes)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[TextExporter] Saved text to: {path}")

        if export_csv:
            path = os.path.join(self.output_dir, f"{base}_multitrack_notes.csv")
            self.exporter.to_csv(merged.notes, path)

        if export_json:
            path = os.path.join(self.output_dir, f"{base}_multitrack.json")
            self.exporter.to_json_multitrack(merged, path)

        if export_song:
            path = os.path.join(self.output_dir, f"{base}_song.json")
            export_minecraft_song(
                merged.notes, tempo, path, self.mono.quantize_subdivisions
            )

        if show_plot or save_plot:
            save_path = None
            if save_plot:
                save_path = os.path.join(self.output_dir, f"{base}_multitrack_dashboard.png")
            self.visualizer.multi_voice_dashboard(
                y_mix, merged.notes, stem_results, save_path=save_path
            )

        return merged


class PolyphonicPipeline:
    """Transcribe a single file with Basic Pitch (no stem separation)."""

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 22050,
        min_confidence: float = 0.4,
        min_note_duration_s: float = 0.05,
        quantize_subdivisions: int = 4,
        output_dir: str = "./output",
    ):
        self.filepath = filepath
        self.mono = MonophonicPipeline(
            filepath=filepath,
            sample_rate=sample_rate,
            min_confidence=min_confidence,
            min_note_duration_s=min_note_duration_s,
            quantize_subdivisions=quantize_subdivisions,
            output_dir=output_dir,
        )

    def run(
        self,
        export_midi: bool = True,
        export_text: bool = True,
        export_csv: bool = False,
        export_json: bool = False,
        export_song: bool = False,
        show_plot: bool = False,
        save_plot: bool = False,
    ) -> TranscriptionResult:
        os.makedirs(self.mono.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.filepath))[0]

        print(f"\n[PolyphonicPipeline] Loading: {self.filepath}")
        y, sr = self.mono.loader.load(self.filepath)
        tempo = self.mono.onset.detect_tempo(y)
        print(f"      Tempo: {tempo:.1f} BPM")

        result = self.mono.transcribe_polyphonic(y, sr, label="mix", tempo=tempo)
        self.mono._apply_result(result)

        if export_midi:
            path = os.path.join(self.mono.output_dir, f"{base}_poly.mid")
            self.mono.midi_writer.write(result.notes, path, tempo_bpm=tempo)

        if export_text:
            path = os.path.join(self.mono.output_dir, f"{base}_poly_notes.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.mono.exporter.to_text(result.notes))
            print(f"[TextExporter] Saved text to: {path}")

        if export_csv:
            self.mono.exporter.to_csv(result.notes, os.path.join(self.mono.output_dir, f"{base}_poly_notes.csv"))

        if export_json:
            self.mono.exporter.to_json(result.notes, os.path.join(self.mono.output_dir, f"{base}_poly.json"))

        if export_song:
            path = os.path.join(self.mono.output_dir, f"{base}_song.json")
            export_minecraft_song(
                result.notes, tempo, path, self.mono.quantize_subdivisions
            )

        if show_plot or save_plot:
            save_path = os.path.join(self.mono.output_dir, f"{base}_poly_dashboard.png") if save_plot else None
            self.mono.visualizer.piano_roll(result.notes, title="Polyphonic Piano Roll", save_path=save_path)

        return result


class StemsPolyPipeline(StemPipeline):
    """Alias for stem mode with per-stem poly configurable via stem_modes."""

    pass


# Backward compatibility
TranscriptionPipeline = MonophonicPipeline
