# WhisperFlow-local

Fully local dictation + meeting transcription for **Windows, macOS, and
Linux**. No audio ever leaves the machine. The engine auto-selects the best
backend for the hardware:

1. **NVIDIA GPU** → faster-whisper on CUDA (float16)
2. **Apple silicon / AMD / Intel GPU** → whisper.cpp (`pywhispercpp`) when
   installed (Metal on macOS; Vulkan builds for AMD/Intel)
3. **No GPU** → faster-whisper on CPU (int8)

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+D` | Toggle dictation. Beep = recording; press again to stop. Text is transcribed and **pasted into whatever window has focus** (Cmd+V on macOS). |
| `Ctrl+Alt+M` | Toggle meeting recording. Captures your **mic** and the **system audio** (the other participants in Teams/Zoom/Webex/etc.). On stop, writes a merged timestamped transcript. Multiple remote speakers are told apart by voice (local diarization) and labeled `Them-1`, `Them-2`, ... |
| `Ctrl+C` (in the console) | Quit. |

## Installing on a new machine

Copy this folder (or clone it), then:

- **Windows:** `powershell -ExecutionPolicy Bypass -File install.ps1`
- **macOS / Linux:** `bash install.sh`

The installer detects NVIDIA GPUs (adds CUDA libraries) and on macOS adds the
whisper.cpp backend for Metal acceleration. Afterwards run **`whisperflow`**
from any terminal (`whisperflow --list` shows audio devices). Config and data
live in `~/.whisperflow/`. First run downloads the model (~1.6GB).

On *this* machine (the original dev setup) keep using `WhisperFlow.bat` —
config and data stay in this folder.

### Platform notes

- **Windows:** pasting into apps that run *as Administrator* requires running
  WhisperFlow elevated too. Meeting mode records the default output via WASAPI
  loopback — no setup needed.
- **macOS:** grant your terminal **Accessibility** and **Input Monitoring**
  permissions (System Settings → Privacy & Security) or hotkeys/paste won't
  work. For meeting system-audio, install
  [BlackHole](https://existential.audio/blackhole/), create a Multi-Output
  Device (speakers + BlackHole) in Audio MIDI Setup, and select it as the
  sound output during meetings.
- **Linux:** needs X11 or XWayland for global hotkeys, and `xclip` (or `xsel`)
  for clipboard paste. Meeting mode records the PulseAudio/PipeWire *monitor*
  source automatically.

## Output

- Dictations append to `history/dictations.jsonl` (and stay in the clipboard).
- Meetings go to `meetings/<date_time>/` containing `mic.wav`, `system.wav`,
  and `transcript.md` with lines like `[00:03:12] Me: ...` / `[00:03:15] Them-2: ...`.

## Subtitles for VLC (`subtitles.py`)

Generate an `.srt` subtitle file from a local video (or audio) file, in the
source language or translated to English, then watch in VLC with subtitles
on:

```
python subtitles.py movie.mkv                # subtitles in the source language
python subtitles.py movie.mkv --translate     # English subtitles, any source language
python subtitles.py movie.mkv --language ja   # skip language auto-detect
python subtitles.py movie.mkv -o out.srt      # custom output path
```

By default the `.srt` is written next to the video with the same base name
(`movie.mkv` -> `movie.srt`) — VLC auto-loads a subtitle file matching that
convention as soon as you open the video, no extra setup needed. Otherwise
use VLC's **Subtitle -> Add Subtitle File...** and point it at the output.

Notes:

- This is a batch step, not a live captioner — Whisper needs a full pass over
  the audio to keep timestamps and accuracy solid, so run it before hitting
  play. A movie-length file still finishes well ahead of watch time on a GPU.
- `--translate` uses Whisper's built-in translate task, which only targets
  **English**. Translating into another target language would need a
  separate machine-translation step on top of the transcript.
- Video containers (mp4, mkv, ...) are decoded directly — no `ffmpeg` binary
  required; faster-whisper/PyAV read the audio track on their own.
- If installed as a package (`pip install .`), the same tool is available as
  `whisper-subtitles`.

## Live captions / live translation (`live_captions.py`)

A floating always-on-top caption bar for whatever's currently playing (a
video in VLC, a stream, a call) — captures the same system-audio loopback
source meeting mode uses, so there's no VLC-side setup:

```
python live_captions.py                # live captions, source language
python live_captions.py --translate    # live English captions, any source language
python live_captions.py --language ja  # skip language auto-detect
python live_captions.py --no-overlay   # print to the console instead of a window
```

Play the video as normal; the overlay floats on top of it like a caption
bar. Press **Esc** with the overlay focused, or Ctrl+C in the console, to
stop.

How it works — a rolling buffer, not a batch pass:

- System audio is captured continuously into a buffer.
- Every `--step` seconds (default 1.2s) the buffer-so-far is re-transcribed;
  the tentative text replaces the on-screen caption right away, so it
  visibly fills in as the speaker talks.
- Once a trailing pause is detected, that line is finalized and the buffer
  clears for the next one.
- If someone talks on with no pause, the buffer is capped at `--window`
  seconds (default 10) and force-flushed so latency and per-pass cost stay
  bounded.

This re-transcribes the growing buffer from scratch each pass rather than
doing true incremental decoding, so there's a real latency floor set by
`--step` and how fast the model runs on your hardware. **Model choice
matters a lot more here than for batch subtitles** — `large-v3-turbo` on a
good NVIDIA GPU is fine, but on CPU or a modest GPU drop to `distil-large-v3`
or `small` in `config.json`, or captions will visibly lag the audio.

`--translate` has the same English-only limitation as `subtitles.py`'s
`--translate`.

## Config (`config.json`)

- `model` — any faster-whisper model id. `large-v3-turbo` is the default; use
  `distil-large-v3` (English-only) or `small` for fast CPU-only machines.
- `language` — `"auto"` or a code like `"en"` (fixing it skips detection).
- `backend` / `device` / `compute_type` — all default `"auto"` (rules at the
  top). Force e.g. `"device": "cpu"` or `"backend": "whisper-cpp"`.
- `dictation_hotkey` / `meeting_hotkey` — e.g. `"ctrl+alt+d"`, `"win+shift+f9"`.
- `mic_device` — `null` = OS default mic. Or a device index / name substring
  (e.g. `"Jabra"`, `"DJI"`, `"C920"`); `--list` shows what's available.
- `system_audio_device` — override the meeting-mode system-audio source by
  name (rarely needed; auto-detected per platform).
- `custom_words` — names/jargon fed to the model as a hint
  (e.g. `["Gordon Technologies", "Redmine"]`).
- `diarize` — tell remote speakers apart in meeting transcripts (default on;
  first use downloads an ~80MB speaker-embedding model).
- `diarize_threshold` — voice-similarity cutoff. If two people merge into one
  speaker, lower it (0.5); if one person splits into several, raise it (0.7).
- `restore_clipboard` — `true` restores your previous clipboard after pasting.
- `beeps` — audio feedback on/off.

Changes take effect on restart.

## Hardware guidance

| Machine | Works? | Suggested config |
|---|---|---|
| NVIDIA GPU (Win/Linux) | best | defaults (CUDA auto-detected) |
| Apple silicon Mac | great | install.sh adds whisper.cpp → Metal GPU |
| AMD/Intel GPU | good | install a Vulkan `pywhispercpp` build, else CPU |
| CPU only | fine | `"model": "distil-large-v3"` or `"small"` |
