"""
main.py — CLI entry point for the monophonic melody transcriber.

Usage:
    python main.py <audio_file> [options]

Examples:
    python main.py song.mp3
    python main.py song.wav --output-dir ./results --no-plot
    python main.py song.mp3 --export-formats midi csv json text
    python main.py song.mp3 --info-only

Dependencies:
    pip install librosa numpy matplotlib mido soundfile
"""

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monophonic Melody Transcriber — converts audio to MIDI/notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py guitar_solo.mp3
  python main.py flute.wav --output-dir ./out --no-plot
  python main.py melody.mp3 --export-formats midi json
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to audio file (.mp3, .wav, .flac, .ogg)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory to save output files (default: ./output)"
    )
    parser.add_argument(
        "--export-formats", "-f",
        nargs="+",
        choices=["midi", "csv", "json", "text"],
        default=["midi", "text"],
        help="Output formats to export (default: midi text)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization (useful for headless/server environments)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as PNG files instead of showing them interactively"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=22050,
        help="Target sample rate for audio loading (default: 22050)"
    )
    parser.add_argument(
        "--confidence-threshold", "-c",
        type=float,
        default=0.4,
        help="Minimum confidence to keep a detected note (default: 0.4)"
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16],
        help="Quantization grid: 4=sixteenth notes, 8=thirty-second notes (default: 4)"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show audio file info, don't run transcription"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )

    return parser.parse_args()


def check_dependencies():
    """Check that required packages are installed and warn about optional ones."""
    missing_required = []
    missing_optional = []

    try:
        import numpy
    except ImportError:
        missing_required.append("numpy")

    try:
        import librosa
    except ImportError:
        missing_required.append("librosa")

    try:
        import matplotlib
    except ImportError:
        missing_optional.append("matplotlib (needed for visualization)")

    try:
        import mido
    except ImportError:
        missing_optional.append("mido (needed for MIDI export)")

    if missing_required:
        print("ERROR: Missing required packages:")
        for pkg in missing_required:
            print(f"  pip install {pkg}")
        sys.exit(1)

    if missing_optional:
        print("WARNING: Missing optional packages (some features unavailable):")
        for pkg in missing_optional:
            print(f"  pip install {pkg.split(' ')[0]}")
        print()


def print_banner():
    print("=" * 55)
    print("  Monophonic Melody Transcriber")
    print("  Audio → Pitch Detection → MIDI")
    print("=" * 55)


def print_summary(notes, output_files, elapsed_s):
    from output.text_export import TextExporter
    exporter = TextExporter()
    stats = exporter.summary(notes)

    print("\n" + "=" * 55)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 55)
    print(f"  Time taken:       {elapsed_s:.2f}s")
    print(f"  Notes detected:   {stats.get('total_notes', 0)}")
    print(f"  Song duration:    {stats.get('total_duration_s', 0):.2f}s")

    pr = stats.get("pitch_range", {})
    if pr:
        print(f"  Pitch range:      {pr.get('lowest')} → {pr.get('highest')} ({pr.get('span_semitones')} semitones)")

    dur = stats.get("duration", {})
    if dur:
        print(f"  Note duration:    {dur.get('shortest_ms')}ms – {dur.get('longest_ms')}ms avg")

    conf = stats.get("confidence", {})
    if conf:
        low = conf.get('low_confidence_notes', 0)
        print(f"  Avg confidence:   {conf.get('average', 0):.2f}  ({low} low-confidence notes)")

    print("\nOutput files:")
    for path in output_files:
        print(f"  {path}")
    print("=" * 55)


def main():
    args = parse_args()
    print_banner()
    check_dependencies()

    # --- validate input file ---
    if not os.path.exists(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    supported = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff']
    ext = os.path.splitext(args.audio_file)[1].lower()
    if ext not in supported:
        print(f"WARNING: Unsupported extension '{ext}'. Supported: {supported}")

    # --- imports (after dependency check) ---
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
    import librosa

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]

    start_time = time.time()

    # ------------------------------------------------------------------ #
    # STEP 1 — Load audio
    # ------------------------------------------------------------------ #
    print(f"\n[1/7] Loading audio: {args.audio_file}")
    loader = AudioLoader(target_sr=args.sample_rate)
    y, sr = loader.load(args.audio_file)

    # print basic file info
    duration = len(y) / sr
    print(f"      Duration: {duration:.2f}s  |  Sample rate: {sr} Hz  |  Samples: {len(y):,}")

    if args.info_only:
        seg = Segmenter(sr=sr)
        info = seg.info(y)
        print("\nSegmenter info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # STEP 2 — Preprocess
    # ------------------------------------------------------------------ #
    print("[2/7] Preprocessing...")
    pre = Preprocessor()
    y = pre.normalize(y)
    y = pre.trim_silence(y, sr)
    y = pre.apply_preemphasis(y)
    if args.verbose:
        print(f"      After trimming: {len(y)/sr:.2f}s")

    # ------------------------------------------------------------------ #
    # STEP 3 — Detect pitch
    # ------------------------------------------------------------------ #
    print("[3/7] Detecting pitch (pYIN algorithm)...")
    detector = PitchDetector(sr=sr)
    f0, voiced_flag = detector.detect(y)
    times = librosa.times_like(f0, sr=sr)
    voiced_count = voiced_flag.sum()
    print(f"      {voiced_count}/{len(voiced_flag)} frames voiced ({100*voiced_count/len(voiced_flag):.1f}%)")

    # ------------------------------------------------------------------ #
    # STEP 4 — Detect tempo
    # ------------------------------------------------------------------ #
    print("[4/7] Detecting tempo...")
    onset_det = OnsetDetector(sr=sr)
    tempo = onset_det.detect_tempo(y)
    print(f"      Detected tempo: {tempo:.1f} BPM")

    # ------------------------------------------------------------------ #
    # STEP 5 — Build + clean notes
    # ------------------------------------------------------------------ #
    print("[5/7] Building note list...")
    builder = NoteBuilder()
    notes = builder.build(f0, voiced_flag, times)
    print(f"      Raw notes detected: {len(notes)}")

    cleaner = NoteCleaner(
        min_duration_s=0.05,
        min_confidence=args.confidence_threshold
    )
    notes = cleaner.clean(notes)
    print(f"      After cleaning: {len(notes)} notes")

    # ------------------------------------------------------------------ #
    # STEP 6 — Quantize
    # ------------------------------------------------------------------ #
    print(f"[6/7] Quantizing to grid (subdivisions={args.subdivisions})...")
    quantizer = Quantizer(tempo=tempo, subdivisions=args.subdivisions)
    notes = quantizer.snap_to_grid(notes)

    # ------------------------------------------------------------------ #
    # STEP 7 — Export
    # ------------------------------------------------------------------ #
    print("[7/7] Exporting results...")
    output_files = []
    exporter = TextExporter()
    writer = MidiWriter()

    if "midi" in args.export_formats:
        path = os.path.join(args.output_dir, f"{base_name}.mid")
        writer.write(notes, path, tempo_bpm=tempo)
        output_files.append(path)

    if "text" in args.export_formats:
        path = os.path.join(args.output_dir, f"{base_name}_notes.txt")
        text = exporter.to_text(notes)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[TextExporter] Saved text to: {path}")
        output_files.append(path)

    if "csv" in args.export_formats:
        path = os.path.join(args.output_dir, f"{base_name}_notes.csv")
        exporter.to_csv(notes, path)
        output_files.append(path)

    if "json" in args.export_formats:
        path = os.path.join(args.output_dir, f"{base_name}_notes.json")
        exporter.to_json(notes, path)
        output_files.append(path)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    if not args.no_plot:
        print("\nGenerating visualizations...")
        viz = Visualizer(sr=sr)
        save_path = None

        if args.save_plots:
            save_path = os.path.join(args.output_dir, f"{base_name}_dashboard.png")
            output_files.append(save_path)

        viz.full_dashboard(y, f0, voiced_flag, notes, save_path=save_path)

    elapsed = time.time() - start_time
    print_summary(notes, output_files, elapsed)


if __name__ == "__main__":
    main()
