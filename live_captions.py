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
it over the video like a caption bar (click and drag it anywhere; Esc or
Ctrl+C to stop).

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

--- Pre-buffered mode for a LOCAL FILE (--file), ahead of playback --------

If you have the video as a file (not a genuine live stream), use --file
instead: it decodes and transcribes the file directly, as fast as the
model can go -- typically faster than real-time on a GPU -- so segments for
a moment in the video are ready *before* playback reaches it. Captions then
appear with none of the rolling-buffer's tentative-text lag, at the exact
timestamp Whisper gave them.

    python live_captions.py --file movie.mkv
    python live_captions.py --file movie.mkv --translate

Exact steps:
  1. Run the command above. Do NOT press Play in VLC yet.
  2. It starts pre-transcribing the file in the background immediately, and
     prints progress. Wait for: "Waiting for playback to start ..."
  3. Open the video in VLC and press Play. The moment audio is heard on your
     speakers, that instant becomes the sync point ("Playback detected").
  4. Captions now appear in sync with the video automatically.

If the on-screen captions are consistently a bit early or late (loopback
detection isn't frame-accurate), add --offset to correct it:
  --offset 0.5    caption everything 0.5s later
  --offset -0.5   caption everything 0.5s earlier

If your system doesn't have a working loopback/monitor source for step 3
(detection fails), use --start-now instead: it skips audio detection and
starts the sync clock the instant you press Enter in the console, so start
VLC playback in the same motion as pressing Enter.
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
    fmt_clock,
    load_config,
    log,
    resolve_system_audio_source,
    to_whisper_audio,
)

SILENCE_TAIL_SEC = 0.8      # how much trailing audio to check for a pause
SILENCE_RMS = 0.008         # below this RMS (float32, -1..1) counts as silence
MIN_FINALIZE_SEC = 0.6      # don't finalize on a near-empty blip
FLUSH_OVERLAP_SEC = 1.0     # audio kept across a forced flush, for continuity


class SegmentTimeline:
    """Time-ordered (start, end, text) segments from pre-transcribing a
    whole file, filled progressively by a background thread and drained by
    the display loop as playback reaches each segment's start time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._segments = []
        self._next = 0
        self.done = threading.Event()  # set once pre-transcription finishes
        self.error = None

    def add(self, start, end, text):
        with self._lock:
            self._segments.append((start, end, text))

    def due(self, elapsed):
        """Segments whose start time has arrived, not yet returned, oldest first."""
        out = []
        with self._lock:
            while self._next < len(self._segments) and self._segments[self._next][0] <= elapsed:
                out.append(self._segments[self._next])
                self._next += 1
        return out

    def has_pending(self):
        """True if there are segments not yet returned by due() -- regardless
        of whether their start time has arrived. Non-destructive: safe to
        call just to check, unlike due() which consumes what it returns."""
        with self._lock:
            return self._next < len(self._segments)

    def lead_seconds(self, elapsed):
        """How far ahead (or behind, if negative) transcription is of
        `elapsed`, based on the newest segment seen so far."""
        with self._lock:
            if not self._segments:
                return 0.0
            return self._segments[-1][1] - elapsed


def pretranscribe_file(engine, task, path, timeline, stop_event):
    """Background thread body: transcribe `path` end-to-end, filling
    `timeline` as segments become available. Typically runs faster than
    real-time on a GPU, so this gets ahead of wherever playback is."""
    t0 = time.time()
    last_end = 0.0
    try:
        for start, end, text in engine.transcribe_stream(str(path), task=task):
            if stop_event.is_set():
                return
            timeline.add(start, end, text)
            last_end = end
    except Exception as e:
        timeline.error = e
    finally:
        timeline.done.set()
        took = time.time() - t0
        log(f"Pre-transcription complete: {fmt_clock(last_end)} of audio in "
            f"{fmt_clock(took)} -- ready and waiting on playback.")


def wait_for_playback_start(cfg):
    """Block until system-audio loopback picks up non-silent audio, i.e.
    until the video is actually playing -- used as the sync point (t0) for
    --file mode. Returns time.time() at that instant."""
    buf = RollingAudioBuffer()
    close_capture, source = open_capture(cfg, buf)
    log(f"Waiting for playback to start (listening on '{source}') ...")
    log(">>> Open the video in VLC and press Play now. <<<")
    try:
        while True:
            time.sleep(0.1)
            audio = buf.snapshot()
            tail = audio[-int(0.2 * SAMPLE_RATE):]
            if len(tail) and float(np.sqrt(np.mean(tail ** 2))) > SILENCE_RMS:
                return time.time()
    finally:
        close_capture()


def file_caption_loop(timeline, t0, offset, emit, stop_event, hide_after=4.0):
    """Displays timeline segments as wall-clock time (since t0, the moment
    playback started) reaches each one's start -- so captions land exactly
    when Whisper says the words happen, not when they finish transcribing."""
    shown_until = None
    while not stop_event.is_set():
        time.sleep(0.15)
        # +offset delays display (needs more wall time to become "due"),
        # -offset shows captions earlier -- matches the --offset help text
        elapsed = time.time() - t0 - offset
        for _start, end, text in timeline.due(elapsed):
            emit(text, final=True)
            shown_until = end
        if shown_until is not None and elapsed > shown_until + hide_after:
            emit("", final=True)
            shown_until = None
        if (timeline.done.is_set() and not timeline.has_pending()
                and shown_until is None):
            # pre-transcription is fully done, every segment has already
            # been shown and hidden -- nothing left will ever become due
            return


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

        # borderless window has no title bar to grab -- drag from anywhere on it
        self._drag = None
        for widget in (self.root, self.label):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.root.bind("<Escape>", lambda _e: self.stop())
        self._poll()

    def _drag_start(self, event):
        self._drag = (event.x_root, event.y_root,
                       self.root.winfo_x(), self.root.winfo_y())

    def _drag_move(self, event):
        start_x, start_y, win_x, win_y = self._drag
        x = win_x + (event.x_root - start_x)
        y = win_y + (event.y_root - start_y)
        self.root.geometry(f"+{x}+{y}")

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


def make_emit(overlay):
    def emit(text, final):
        if overlay:
            overlay.set_text(text)
        else:
            end = "\n" if final else "\r"
            print(f"{text}{' ' * 20}", end=end, flush=True)
    return emit


def run_live_mode(args, cfg, engine, task):
    """Rolling-buffer captions of whatever's currently playing (loopback)."""
    audio_buf = RollingAudioBuffer()
    try:
        close_capture, source = open_capture(cfg, audio_buf)
    except Exception as e:
        sys.exit(f"error: could not open system audio ({e})")
    log(f"Live captions: capturing '{source}' (task={task}). Ctrl+C to stop.")

    overlay = None if args.no_overlay else CaptionOverlay()
    emit = make_emit(overlay)

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


def run_file_mode(args, cfg, engine, task):
    """Pre-transcribe a local file ahead of playback, then display each
    segment exactly when playback reaches it. See the module docstring for
    the exact step-by-step sequence."""
    stop_event = threading.Event()
    timeline = SegmentTimeline()
    pretranscribe = threading.Thread(
        target=pretranscribe_file,
        args=(engine, task, args.file, timeline, stop_event),
        daemon=True,
    )
    log(f"Pre-transcribing '{args.file}' in the background (task={task}) ...")
    pretranscribe.start()

    if args.start_now:
        input(">>> Press Enter the INSTANT you click Play in VLC <<< ")
        t0 = time.time()
    else:
        t0 = wait_for_playback_start(cfg)
        log("Playback detected -- captions now syncing to the video.")

    overlay = None if args.no_overlay else CaptionOverlay()
    emit = make_emit(overlay)

    display_worker = threading.Thread(
        target=file_caption_loop,
        args=(timeline, t0, args.offset, emit, stop_event),
        daemon=True,
    )
    display_worker.start()

    try:
        if overlay:
            overlay.run()
        else:
            while display_worker.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if timeline.error:
            log(f"!! Pre-transcription error: {timeline.error}")
        print()
        log("Live captions: stopped")


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
    ap.add_argument("--window", type=float, default=10.0, help="max seconds of audio kept in the rolling buffer before a forced flush (default 10, ignored with --file)")
    ap.add_argument("--step", type=float, default=1.2, help="seconds between re-transcribe passes (default 1.2; lower = snappier but costs more; ignored with --file)")
    ap.add_argument(
        "--file",
        help="pre-transcribe this local video/audio file ahead of playback "
        "instead of live rolling-buffer captions -- see the module "
        "docstring (top of live_captions.py) for the exact steps",
    )
    ap.add_argument(
        "--offset", type=float, default=0.0,
        help="shift --file captions by this many seconds (+later, -earlier) "
        "to correct for imprecise playback-start detection (default 0)",
    )
    ap.add_argument(
        "--start-now", action="store_true",
        help="with --file: skip loopback-based playback detection and start "
        "the sync clock the instant you press Enter, instead of listening "
        "for audio to begin",
    )
    args = ap.parse_args()

    cfg = load_config()
    if args.language:
        cfg["language"] = args.language
    engine = Engine(cfg)
    task = "translate" if args.translate else "transcribe"

    if args.file:
        run_file_mode(args, cfg, engine, task)
    else:
        run_live_mode(args, cfg, engine, task)


if __name__ == "__main__":
    main()
