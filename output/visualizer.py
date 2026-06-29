import numpy as np
from models.note import Note

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class Visualizer:
    """
    Produces visualizations of audio signals and transcribed notes.

    Three main views:
      1. Waveform       — amplitude over time (time domain)
      2. Spectrogram    — frequency content over time (frequency domain)
      3. Piano Roll     — detected notes as colored rectangles over time
    """

    def __init__(self, sr: int = 22050, figsize: tuple = (14, 4)):
        self.sr = sr
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Waveform
    # ------------------------------------------------------------------

    def waveform(self, y: np.ndarray, title: str = "Waveform", save_path: str = None):
        """
        Plot the raw audio waveform — amplitude vs time.

        What you're seeing:
          - X axis: time in seconds
          - Y axis: amplitude (-1.0 to 1.0)
          - Loud sounds = tall peaks
          - Silence = flat line near zero
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Visualizer] matplotlib not installed.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)
        times = np.linspace(0, len(y) / self.sr, len(y))
        ax.plot(times, y, color='steelblue', linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    def spectrogram(self, y: np.ndarray, title: str = "Spectrogram", save_path: str = None):
        """
        Plot a mel spectrogram — frequency content vs time.

        What you're seeing:
          - X axis: time in seconds
          - Y axis: frequency (mel scale, low → high)
          - Color: intensity (bright = loud at that frequency)
          - Horizontal bands = sustained notes
          - Vertical streaks = transients (drum hits, plucks)
        """
        if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
            print("[Visualizer] librosa and matplotlib required.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=self.sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title)
        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Pitch overlay on spectrogram
    # ------------------------------------------------------------------

    def pitch_overlay(
        self,
        y: np.ndarray,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        title: str = "Pitch Detection",
        save_path: str = None
    ):
        """
        Plot spectrogram with detected pitch curve overlaid.
        Voiced frames (where a note is playing) shown in yellow.
        Unvoiced frames (silence/noise) shown as gaps.
        """
        if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
            print("[Visualizer] librosa and matplotlib required.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        # draw spectrogram as background
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=self.sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma', alpha=0.8)

        # overlay pitch curve — only where voiced
        times = librosa.times_like(f0, sr=self.sr)
        f0_voiced = np.where(voiced_flag, f0, np.nan)  # NaN where unvoiced → gaps in plot
        ax.plot(times, f0_voiced, color='yellow', linewidth=1.5, label='Detected pitch')
        ax.legend(loc='upper right')
        ax.set_title(title)
        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Piano roll
    # ------------------------------------------------------------------

    def piano_roll(self, notes: list[Note], title: str = "Piano Roll", save_path: str = None):
        """
        Plot detected notes as a piano roll — the standard way to visualize MIDI.

        What you're seeing:
          - X axis: time in seconds
          - Y axis: MIDI pitch (low = bottom, high = top)
          - Each rectangle = one note
          - Rectangle width = note duration
          - Color intensity = confidence
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Visualizer] matplotlib not installed.")
            return

        if not notes:
            print("[Visualizer] No notes to display.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        # draw one rectangle per note
        for note in notes:
            color_intensity = 0.3 + 0.7 * note.confidence  # dim low-confidence notes
            rect = mpatches.FancyBboxPatch(
                (note.start_time, note.midi_pitch - 0.4),  # (x, y)
                note.duration,                              # width
                0.8,                                        # height
                boxstyle="round,pad=0.02",
                facecolor=(0.2, 0.5 * color_intensity, color_intensity),
                edgecolor='white',
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)

            # label notes if they're wide enough to fit text
            if note.duration > 0.15:
                ax.text(
                    note.start_time + note.duration / 2,
                    note.midi_pitch,
                    note.note_name,
                    ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold'
                )

        # axes formatting
        pitches = [n.midi_pitch for n in notes]
        ax.set_xlim(0, max(n.end_time for n in notes) + 0.5)
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Pitch")
        ax.set_title(title)

        # add note name labels on Y axis for reference
        y_ticks = range(min(pitches) - 1, max(pitches) + 2)
        if LIBROSA_AVAILABLE:
            import librosa
            ax.set_yticks(list(y_ticks))
            ax.set_yticklabels([librosa.midi_to_note(p) for p in y_ticks], fontsize=7)

        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Full dashboard — all views in one figure
    # ------------------------------------------------------------------

    def full_dashboard(
        self,
        y: np.ndarray,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        notes: list[Note],
        save_path: str = None
    ):
        """
        Show waveform, pitch overlay, and piano roll stacked vertically.
        The best view for understanding the full transcription pipeline.
        """
        if not MATPLOTLIB_AVAILABLE or not LIBROSA_AVAILABLE:
            print("[Visualizer] librosa and matplotlib required.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle("Transcription Dashboard", fontsize=14, fontweight='bold')

        # --- row 1: waveform ---
        times_wave = np.linspace(0, len(y) / self.sr, len(y))
        axes[0].plot(times_wave, y, color='steelblue', linewidth=0.4, alpha=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].axhline(0, color='gray', linewidth=0.4, linestyle='--')

        # --- row 2: spectrogram + pitch ---
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=self.sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma', alpha=0.85)
        times_f0 = librosa.times_like(f0, sr=self.sr)
        f0_voiced = np.where(voiced_flag, f0, np.nan)
        axes[1].plot(times_f0, f0_voiced, color='yellow', linewidth=1.5, label='Pitch')
        axes[1].set_title("Spectrogram + Pitch Curve")
        axes[1].legend(loc='upper right', fontsize=8)

        # --- row 3: piano roll ---
        if notes:
            pitches = [n.midi_pitch for n in notes]
            for note in notes:
                intensity = 0.3 + 0.7 * note.confidence
                rect = mpatches.FancyBboxPatch(
                    (note.start_time, note.midi_pitch - 0.4),
                    note.duration, 0.8,
                    boxstyle="round,pad=0.02",
                    facecolor=(0.2, 0.5 * intensity, intensity),
                    edgecolor='white', linewidth=0.4, alpha=0.85
                )
                axes[2].add_patch(rect)
                if note.duration > 0.15:
                    axes[2].text(
                        note.start_time + note.duration / 2, note.midi_pitch,
                        note.note_name, ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold'
                    )
            axes[2].set_xlim(0, max(n.end_time for n in notes) + 0.5)
            axes[2].set_ylim(min(pitches) - 2, max(pitches) + 2)
            y_ticks = range(min(pitches) - 1, max(pitches) + 2)
            axes[2].set_yticks(list(y_ticks))
            axes[2].set_yticklabels([librosa.midi_to_note(p) for p in y_ticks], fontsize=7)

        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("MIDI Pitch")
        axes[2].set_title("Piano Roll")
        axes[2].grid(axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_or_save(self, fig, save_path: str = None):
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Saved plot to: {save_path}")
            plt.close(fig)
        else:
            plt.show()
