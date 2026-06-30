"""
main.py — CLI entry point for Midicraft transcription.

Usage:
    python main.py <audio_file> [options]

Examples:
    python main.py song.mp3
    python main.py song.mp3 --mode stems --no-plot
    python main.py song.mp3 --mode poly
    python main.py song.mp3 --separate-only
    python main.py song.mp3 --mode stems+poly --stem-modes other:poly
"""

import argparse
import os
import sys
import time


def parse_stem_modes(value: str) -> dict[str, str]:
    """Parse 'vocals:mono,other:poly' into a dict."""
    modes: dict[str, str] = {}
    if not value:
        return modes
    for part in value.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        stem, mode = part.split(":", 1)
        modes[stem.strip()] = mode.strip()
    return modes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Midicraft — audio transcription with mono, stem, and polyphonic modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py guitar_solo.mp3
  python main.py song.mp3 --mode stems --no-plot
  python main.py song.mp3 --separate-only
  python main.py song.mp3 --mode stems+poly --stem-modes other:poly
        """,
    )

    parser.add_argument("audio_file", help="Path to audio file (.mp3, .wav, .flac, .ogg)")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "--export-formats", "-f",
        nargs="+",
        choices=["midi", "csv", "json", "text"],
        default=["midi", "text"],
        help="Output formats (default: midi text)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--sample-rate", "-sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--confidence-threshold", "-c", type=float, default=0.4, help="Min note confidence")
    parser.add_argument("--subdivisions", type=int, default=4, choices=[1, 2, 4, 8, 16], help="Quantization grid")
    parser.add_argument("--info-only", action="store_true", help="Only show audio file info")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    mode_group = parser.add_argument_group("transcription mode")
    mode_group.add_argument(
        "--mode",
        choices=["mono", "stems", "poly", "stems+poly"],
        default="mono",
        help="mono (default), stems, poly, or stems+poly",
    )
    mode_group.add_argument(
        "--separate-only",
        action="store_true",
        help="Only separate stems to WAV files, skip transcription",
    )
    mode_group.add_argument(
        "--stems",
        default="vocals,bass,other",
        help="Comma-separated stems to transcribe (default: vocals,bass,other)",
    )
    mode_group.add_argument("--stem-model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    mode_group.add_argument("--stems-dir", help="Load pre-separated stems from this directory")
    mode_group.add_argument("--device", choices=["cpu", "cuda"], help="Device for Demucs (auto-detect if omitted)")
    mode_group.add_argument(
        "--stem-modes",
        default="",
        help="Per-stem detection mode, e.g. vocals:mono,other:poly",
    )

    return parser.parse_args()


def check_dependencies(mode: str):
    missing_required = []
    missing_optional = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing_required.append("numpy")

    try:
        import librosa  # noqa: F401
    except ImportError:
        missing_required.append("librosa")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing_optional.append("matplotlib (needed for visualization)")

    try:
        import mido  # noqa: F401
    except ImportError:
        missing_optional.append("mido (needed for MIDI export)")

    if mode in ("stems", "stems+poly") or mode == "mono":
        pass

    if missing_required:
        print("ERROR: Missing required packages:")
        for pkg in missing_required:
            print(f"  pip install {pkg}")
        sys.exit(1)

    if missing_optional:
        print("WARNING: Missing optional packages:")
        for pkg in missing_optional:
            print(f"  pip install {pkg.split(' ')[0]}")
        print()


def print_banner(mode: str):
    print("=" * 55)
    print("  Midicraft — Audio Transcription")
    print(f"  Mode: {mode}")
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
        print(f"  Pitch range:      {pr.get('lowest')} -> {pr.get('highest')} ({pr.get('span_semitones')} semitones)")

    voices = sorted({n.voice_id for n in notes})
    if len(voices) > 1 or (voices and voices[0] != 0):
        print(f"  Voices/channels:  {voices}")

    print("\nOutput files:")
    for path in output_files:
        print(f"  {path}")
    print("=" * 55)


def main():
    args = parse_args()
    print_banner(args.mode)
    check_dependencies(args.mode)

    if not os.path.exists(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    if args.info_only:
        from audio.loader import AudioLoader
        from audio.segmenter import Segmenter

        loader = AudioLoader(target_sr=args.sample_rate)
        y, sr = loader.load(args.audio_file)
        seg = Segmenter(sr=sr)
        info = seg.info(y)
        print(f"\nDuration: {len(y)/sr:.2f}s  |  SR: {sr} Hz")
        print("\nSegmenter info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    export_kw = dict(
        export_midi="midi" in args.export_formats,
        export_text="text" in args.export_formats,
        export_csv="csv" in args.export_formats,
        export_json="json" in args.export_formats,
        show_plot=not args.no_plot,
        save_plot=args.save_plots,
    )

    stem_list = [s.strip() for s in args.stems.split(",") if s.strip()]
    stem_modes = parse_stem_modes(args.stem_modes)

    if args.mode == "stems+poly":
        for stem in stem_list:
            stem_modes.setdefault(stem, "mono")
        if "other" in stem_list:
            stem_modes["other"] = "poly"

    notes = []
    output_files = []

    if args.mode == "mono":
        from pipeline import MonophonicPipeline

        pipeline = MonophonicPipeline(
            filepath=args.audio_file,
            sample_rate=args.sample_rate,
            min_confidence=args.confidence_threshold,
            quantize_subdivisions=args.subdivisions,
            output_dir=args.output_dir,
        )
        notes = pipeline.run(**export_kw)

    elif args.mode in ("stems", "stems+poly"):
        from pipeline import StemPipeline

        pipeline = StemPipeline(
            filepath=args.audio_file,
            sample_rate=args.sample_rate,
            min_confidence=args.confidence_threshold,
            quantize_subdivisions=args.subdivisions,
            output_dir=args.output_dir,
            stems=stem_list,
            stem_model=args.stem_model,
            device=args.device,
            stem_modes=stem_modes,
            stems_dir=args.stems_dir,
            separate_only=args.separate_only,
        )
        result = pipeline.run(**export_kw)
        notes = result.notes

        if args.separate_only:
            base = os.path.splitext(os.path.basename(args.audio_file))[0]
            stems_out = args.stems_dir or os.path.join(args.output_dir, f"{base}_stems")
            print(f"\nStems saved to: {stems_out}")
            sys.exit(0)

    elif args.mode == "poly":
        from pipeline import PolyphonicPipeline

        try:
            from basic_pitch.inference import predict  # noqa: F401
        except ImportError:
            print("ERROR: basic-pitch required for --mode poly")
            print("  pip install -r requirements-poly.txt")
            sys.exit(1)

        pipeline = PolyphonicPipeline(
            filepath=args.audio_file,
            sample_rate=args.sample_rate,
            min_confidence=args.confidence_threshold,
            quantize_subdivisions=args.subdivisions,
            output_dir=args.output_dir,
        )
        result = pipeline.run(**export_kw)
        notes = result.notes

    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    for fmt in args.export_formats:
        if fmt == "midi":
            suffix = "" if args.mode == "mono" else f"_{'multitrack' if args.mode in ('stems', 'stems+poly') else 'poly'}"
            output_files.append(os.path.join(args.output_dir, f"{base_name}{suffix}.mid"))
        elif fmt == "text":
            suffix = "_notes" if args.mode == "mono" else f"_{'multitrack_notes' if args.mode in ('stems', 'stems+poly') else 'poly_notes'}"
            output_files.append(os.path.join(args.output_dir, f"{base_name}{suffix}.txt"))
        elif fmt == "csv":
            suffix = "_notes" if args.mode == "mono" else f"_{'multitrack_notes' if args.mode in ('stems', 'stems+poly') else 'poly_notes'}"
            output_files.append(os.path.join(args.output_dir, f"{base_name}{suffix}.csv"))
        elif fmt == "json":
            suffix = "" if args.mode == "mono" else f"_{'multitrack' if args.mode in ('stems', 'stems+poly') else 'poly'}"
            output_files.append(os.path.join(args.output_dir, f"{base_name}{suffix}.json"))

    elapsed = time.time() - start_time
    if notes:
        print_summary(notes, [p for p in output_files if os.path.exists(p)], elapsed)
    else:
        print(f"\nDone in {elapsed:.2f}s (no notes exported).")


if __name__ == "__main__":
    main()
