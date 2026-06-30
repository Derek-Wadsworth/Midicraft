"""
pipeline.py — Orchestrates the full transcription pipeline.

This file wires all the modules together into one callable object.
It is the "conductor" — it doesn't do any processing itself,
it just calls the right component at the right time.

Usage:
    from pipeline import TranscriptionPipeline

    pipeline = TranscriptionPipeline("song.mp3")
    notes = pipeline.run()
"""

import librosa
from audio.loader import AudioLoader
from audio.preprocessor import Preprocessor
from audio.segmenter import Segmenter
from features.pitch import PitchDetector
from features.onset import OnsetDetector
from features.spectrum import SpectrumAnalyzer
from transcription.note_builder import NoteBuilder
from transcription.quantizer import Quantizer
from transcription.cleaner import NoteCleaner
from output.midi_writer import MidiWriter
from output.visualizer import Visualizer
from output.text_export import TextExporter
from models.note import Note


class TranscriptionPipeline:
    """
    End-to-end monophonic melody transcription pipeline.

    Stages:
        1. Load       — read audio file into numpy array
        2. Preprocess — normalize, trim silence, apply preemphasis
        3. Pitch      — detect fundamental frequency per frame (pYIN)
        4. Tempo      — estimate BPM via beat tracking
        5. Build      — convert frame-level pitch to Note objects
        6. Clean      — remove artifacts, merge fragments
        7. Quantize   — snap timings to musical grid
        8. Export     — write MIDI, CSV, JSON, text, and/or plots
    """

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 22050,
        min_confidence: float = 0.4,
        min_note_duration_s: float = 0.05,
        quantize_subdivisions: int = 4,
        output_dir: str = "./output",
    ):
        """
        Args:
            filepath:              Path to the input audio file.
            sample_rate:           Target sample rate (22050 Hz standard).
            min_confidence:        Discard notes below this voiced probability.
            min_note_duration_s:   Discard notes shorter than this (seconds).
            quantize_subdivisions: Grid resolution (4 = sixteenth notes).
            output_dir:            Directory for exported files.
        """
        self.filepath = filepath
        self.output_dir = output_dir

        # instantiate all components
        self.loader      = AudioLoader(target_sr=sample_rate)
        self.preprocessor = Preprocessor()
        self.segmenter   = Segmenter(sr=sample_rate)
        self.pitch       = PitchDetector(sr=sample_rate)
        self.onset       = OnsetDetector(sr=sample_rate)
        self.spectrum    = SpectrumAnalyzer(sr=sample_rate)
        self.builder     = NoteBuilder()
        self.cleaner     = NoteCleaner(
            min_confidence=min_confidence,
            min_duration_s=min_note_duration_s,
        )
        self.quantize_subdivisions = quantize_subdivisions
        self.midi_writer = MidiWriter()
        self.visualizer  = Visualizer(sr=sample_rate)
        self.exporter    = TextExporter()

        # state populated after run()
        self.y: "np.ndarray | None" = None
        self.sr: int = sample_rate
        self.f0: "np.ndarray | None" = None
        self.voiced_flag: "np.ndarray | None" = None
        self.tempo: float = 120.0
        self.notes: list[Note] = []

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(
        self,
        export_midi: bool = True,
        export_text: bool = True,
        export_csv: bool = False,
        export_json: bool = False,
        show_plot: bool = True,
        save_plot: bool = False,
    ) -> list[Note]:
        """
        Run the full pipeline and return the detected notes.

        Args:
            export_midi:  Write a .mid file to output_dir.
            export_text:  Write a human-readable .txt file.
            export_csv:   Write a .csv file.
            export_json:  Write a .json file (used later for Minecraft).
            show_plot:    Display visualization interactively.
            save_plot:    Save visualization as PNG to output_dir.

        Returns:
            List of Note objects (also stored in self.notes).
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        self._step_load()
        self._step_preprocess()
        self._step_detect_pitch()
        self._step_detect_tempo()
        self._step_build_notes()
        self._step_clean()
        self._step_quantize()
        self._step_export(export_midi, export_text, export_csv, export_json)

        if show_plot or save_plot:
            save_path = None
            if save_plot:
                import os
                base = os.path.splitext(os.path.basename(self.filepath))[0]
                save_path = os.path.join(self.output_dir, f"{base}_dashboard.png")
            self.visualizer.full_dashboard(
                self.y, self.f0, self.voiced_flag, self.notes,
                save_path=save_path
            )

        return self.notes

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #

    def _step_load(self):
        print(f"\n[1/7] Loading: {self.filepath}")
        self.y, self.sr = self.loader.load(self.filepath)
        duration = len(self.y) / self.sr
        print(f"      {duration:.2f}s  |  {self.sr} Hz  |  {len(self.y):,} samples")

    def _step_preprocess(self):
        print("[2/7] Preprocessing...")
        self.y = self.preprocessor.normalize(self.y)
        self.y = self.preprocessor.trim_silence(self.y, self.sr)
        self.y = self.preprocessor.apply_preemphasis(self.y)

    def _step_detect_pitch(self):
        print("[3/7] Detecting pitch (pYIN)...")
        self.f0, self.voiced_flag, self.voiced_prob = self.pitch.detect_with_confidence(self.y)
        self.times = librosa.times_like(self.f0, sr=self.sr)
        pct = 100 * self.voiced_flag.sum() / len(self.voiced_flag)
        print(f"      {self.voiced_flag.sum()}/{len(self.voiced_flag)} frames voiced ({pct:.1f}%)")

    def _step_detect_tempo(self):
        print("[4/7] Estimating tempo...")
        self.tempo = self.onset.detect_tempo(self.y)
        print(f"      {self.tempo:.1f} BPM")

    def _step_build_notes(self):
        print("[5/7] Building notes...")
        self.notes = self.builder.build_with_confidence(
            self.f0, self.voiced_flag, self.voiced_prob, self.times
        )
        print(f"      {len(self.notes)} raw notes")

    def _step_clean(self):
        print("[6/7] Cleaning...")
        self.notes = self.cleaner.clean(self.notes)
        print(f"      {len(self.notes)} notes after cleaning")

    def _step_quantize(self):
        print("[7/7] Quantizing...")
        quantizer = Quantizer(tempo=self.tempo, subdivisions=self.quantize_subdivisions)
        self.notes = quantizer.snap_to_grid(self.notes)
        info = quantizer.grid_info()
        print(f"      Grid size: {info['grid_size_ms']:.1f}ms ({info['subdivisions']} subdivisions/beat)")

    def _step_export(self, midi, text, csv, json_export):
        import os
        base = os.path.splitext(os.path.basename(self.filepath))[0]

        if midi:
            path = os.path.join(self.output_dir, f"{base}.mid")
            self.midi_writer.write(self.notes, path, tempo_bpm=self.tempo)

        if text:
            path = os.path.join(self.output_dir, f"{base}_notes.txt")
            content = self.exporter.to_text(self.notes)
            with open(path, 'w') as f:
                f.write(content)
            print(f"[TextExporter] Saved text to: {path}")

        if csv:
            path = os.path.join(self.output_dir, f"{base}_notes.csv")
            self.exporter.to_csv(self.notes, path)

        if json_export:
            path = os.path.join(self.output_dir, f"{base}_notes.json")
            self.exporter.to_json(self.notes, path)

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    def print_notes(self):
        """Print the detected note list to the console."""
        print(self.exporter.to_text(self.notes))

    def summary(self) -> dict:
        """Return summary statistics about the transcription."""
        return self.exporter.summary(self.notes)
