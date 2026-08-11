"""
live_captions.py: real-time captions/translation for whatever is currently
playing (e.g. a video in VLC), shown in a floating always-on-top overlay --
fully on-device, nothing leaves the machine.

    python live_captions.py                 # captions in the source language
    python live_captions.py --translate      # live English captions, any source language
    python live_captions.py --language ja    # skip language auto-detect
    python live_captions.py --no-overlay     # print to the console instead of a window

Play the video in VLC (or anything else) as normal; this captures the same
system-audio loopback source used by meeting mode, so no VLC-side setup is
needed. The overlay floats on top of whatever window has focus -- position
it over the video like a caption bar.

How it works (rolling buffer, not a one-shot batch pass):
  - System audio is continuously captured into a rolling buffer.
  - Every --step seconds the buffer-so-far is re-transcribed; the *tentative*
    result replaces the on-screen caption immediately, so text visibly fills
    in as the speaker talks.
  - Once a trailing silence is detected (a pause), that utterance is
    considered final, printed as a settled caption line, and the buffer is
    cleared for the next one.
  - If someone talks for a long stretch with no pause, the buffer is capped
    at --window seconds and force-flushed so latency and per-pass
    transcription cost don't grow unbounded.

This re-transcribes the growing buffer from scratch each pass (simple, and
accurate) rather than doing incremental/streaming decoding, so there's a
real latency floor set by --step and how fast the model runs on your
hardware. For live use, a smaller/faster model than the large default
matters a lot more here than for batch subtitles -- see README.md.
"""
import argparse
import queue
import sys
import threading
import time

import numpy as np

from whisperflow import (
    SAMPLE_RATE,
    Engine,
    IS_WIN,
    load_config,
    log,
    resolve_system_audio_source,
    to_whisper_audio,
)

SILENCE_TAIL_SEC = 0.8      # how much trailing audio to check for a pause
SILENCE_RMS = 0.008         # below this RMS (float32, -1..1) counts as silence
MIN_FINALIZE_SEC = 0.6      # don't finalize on a near-empty blip
FLUSH_OVERLAP_SEC = 1.0     # audio kept across a forced flush, for continuity


class RollingAudioBuffer:
    """Accumulates captured system audio, resampled to 16 kHz mono. A lock
    keeps the capture callback (audio thread) and the transcription loop
    (main thread) from touching the same array at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = []  # float32 mono 16k arrays, oldest first

    def push(self, chunk, rate):
        mono16k = to_whisper_audio(chunk, rate)
        with self._lock:
            self._chunks.append(mono16k)

    def snapshot(self):
        """Current buffer as one array, without clearing it."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            buf = np.concatenate(self._chunks)
            self._chunks = [buf]  # coalesce so the chunk list doesn't grow forever
            return buf

    def drop_seconds(self, seconds):
        """Discard the oldest `seconds` of buffered audio."""
        n = int(seconds * SAMPLE_RATE)
        with self._lock:
            if not self._chunks:
                return
            buf = np.concatenate(self._chunks)
            self._chunks = [buf[n:]] if n < len(buf) else []


def open_capture(cfg, buf):
    """Open the system-audio loopback source and feed it into `buf`.
    Mirrors Meeting._open_system_capture's per-platform stream setup, but
    pushes into a RollingAudioBuffer instead of writing a wav file.
    Returns (close_fn, source_name).
    """
    if IS_WIN:
        import pyaudiowpatch as pyaudio

        pa, speakers = resolve_system_audio_source(cfg)
        rate = int(speakers["defaultSampleRate"])
        channels = max(1, int(speakers["maxInputChannels"]))

        def cb(in_data, frame_count, time_info, status):
            arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                arr = arr.reshape(-1, channels)
            buf.push(arr, rate)
            return (in_data, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
            input_device_index=speakers["index"], frames_per_buffer=2048,
            stream_callback=cb,
        )

        def close():
            stream.stop_stream()
            stream.close()
            pa.terminate()

        return close, speakers["name"]

    import sounddevice as sd

    dev, info = resolve_system_audio_source(cfg)
    rate = int(info["default_samplerate"])
    channels = max(1, min(2, info["max_input_channels"]))

    def cb(indata, *_):
        buf.push(indata.copy(), rate)

    stream = sd.InputStream(
        device=dev, samplerate=rate, channels=channels,
        dtype="float32", callback=cb,
    )
    stream.start()

    def close():
        stream.stop()
        stream.close()

    return close, info["name"]


class CaptionOverlay:
    """Borderless, always-on-top, semi-transparent caption bar (tkinter,
    stdlib only). Floats over whatever's playing, e.g. VLC."""

    def __init__(self):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.85)
        except Exception:
            pass  # not supported on some Linux WMs; window still works
        self.root.configure(bg="black")

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = int(sw * 0.8), 100
        x, y = (sw - w) // 2, sh - h - 70
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.label = tk.Label(
            self.root, text="", fg="white", bg="black",
            font=("Segoe UI", 20, "bold"), wraplength=w - 40, justify="center",
        )
        self.label.pack(expand=True, fill="both")

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.root.bind("<Escape>", lambda _e: self.stop())
        self._poll()

    def set_text(self, text):
        self._queue.put(text)

    def _poll(self):
        try:
            while True:
                text = self._queue.get_nowait()
                self.label.configure(text=text)
        except queue.Empty:
            pass
        if self._stop_event.is_set():
            self.root.quit()
            return
        self.root.after(100, self._poll)

    def run(self):
        self.root.mainloop()

    def stop(self):
        self._stop_event.set()


def caption_loop(engine, task, buf, window_sec, step_sec, emit, stop_event):
    """Runs until stop_event is set. Calls emit(text, final) whenever the
    on-screen caption should change."""
    last_shown = ""
    while not stop_event.is_set():
        time.sleep(step_sec)
        audio = buf.snapshot()
        dur = len(audio) / SAMPLE_RATE
        if dur < 0.3:
            continue

        tail = audio[-int(SILENCE_TAIL_SEC * SAMPLE_RATE):]
        is_paused = len(tail) > 0 and float(np.sqrt(np.mean(tail ** 2))) < SILENCE_RMS

        try:
            segments = engine.transcribe(audio, task=task)
        except Exception as e:
            log(f"!! Live transcription failed: {e}")
            continue
        text = " ".join(s[2] for s in segments).strip()

        if is_paused and dur >= MIN_FINALIZE_SEC:
            if text:
                emit(text, final=True)
            buf.drop_seconds(dur)
            last_shown = ""
        elif dur >= window_sec:
            # long stretch with no pause: force a flush so latency/cost stay
            # bounded, keeping a short overlap for continuity into the next line
            if text:
                emit(text, final=True)
            buf.drop_seconds(dur - FLUSH_OVERLAP_SEC)
            last_shown = ""
        elif text and text != last_shown:
            emit(text, final=False)
            last_shown = text


def main():
    ap = argparse.ArgumentParser(
        description="Live captions/translation overlay for whatever is "
        "currently playing (e.g. a video in VLC), fully on-device.",
    )
    ap.add_argument(
        "--translate", action="store_true",
        help="caption in English regardless of the spoken language "
        "(Whisper's built-in translate task; English-only target)",
    )
    ap.add_argument("--language", help="force the source language (e.g. 'en', 'ja'); default is auto-detect")
    ap.add_argument("--no-overlay", action="store_true", help="print captions to the console instead of a floating window")
    ap.add_argument("--window", type=float, default=10.0, help="max seconds of audio kept in the rolling buffer before a forced flush (default 10)")
    ap.add_argument("--step", type=float, default=1.2, help="seconds between re-transcribe passes (default 1.2; lower = snappier but costs more)")
    args = ap.parse_args()

    cfg = load_config()
    if args.language:
        cfg["language"] = args.language
    engine = Engine(cfg)
    task = "translate" if args.translate else "transcribe"

    audio_buf = RollingAudioBuffer()
    try:
        close_capture, source = open_capture(cfg, audio_buf)
    except Exception as e:
        sys.exit(f"error: could not open system audio ({e})")
    log(f"Live captions: capturing '{source}' (task={task}). Ctrl+C to stop.")

    overlay = None if args.no_overlay else CaptionOverlay()

    def emit(text, final):
        if overlay:
            overlay.set_text(text)
        else:
            end = "\n" if final else "\r"
            print(f"{text}{' ' * 20}", end=end, flush=True)

    stop_event = threading.Event()
    worker = threading.Thread(
        target=caption_loop,
        args=(engine, task, audio_buf, args.window, args.step, emit, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        if overlay:
            overlay.run()
        else:
            while worker.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        close_capture()
        print()
        log("Live captions: stopped")


if __name__ == "__main__":
    main()
