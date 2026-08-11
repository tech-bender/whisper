"""
WhisperFlow-local: hotkey dictation + meeting recording, fully on-device.
Runs on Windows, macOS, and Linux.

  Ctrl+Alt+D  toggle dictation  -> transcribe -> paste into active window
  Ctrl+Alt+M  toggle meeting recording (mic + system audio) -> transcript
  Ctrl+C      quit (in this console)

All audio and transcripts stay on this machine.
"""

import json
import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# ---------------------------------------------------------------------------
# Make pip-installed CUDA libraries (cublas/cudnn) visible to ctranslate2
# before faster_whisper is imported. No-op on machines without them.
# ---------------------------------------------------------------------------
def _preload_nvidia_libs():
    import site

    candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    if hasattr(site, "getsitepackages"):
        candidates.extend(site.getsitepackages())
    for sp in candidates:
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        if IS_WIN:
            for dll_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
        else:
            import ctypes
            for pattern in ("cublas/lib/libcublas.so*", "cudnn/lib/libcudnn.so*"):
                for so in sorted(nvidia.glob(pattern)):
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


_preload_nvidia_libs()

import numpy as np  # noqa: E402
import pyperclip  # noqa: E402
import sounddevice as sd  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

try:
    from pynput import keyboard as pk
    PYNPUT_ERROR = None
except Exception as e:  # e.g. headless Linux without X11
    pk = None
    PYNPUT_ERROR = e

# Config and data live next to the script when running from a source checkout
# (this machine's setup), otherwise in ~/.whisperflow (installed package).
_SOURCE_DIR = Path(__file__).resolve().parent
if (_SOURCE_DIR / "config.json").exists():
    BASE_DIR = _SOURCE_DIR
else:
    BASE_DIR = Path.home() / ".whisperflow"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "model": "large-v3-turbo",
    "backend": "auto",       # auto | faster-whisper | whisper-cpp
    "device": "auto",        # auto | cuda | cpu
    "compute_type": "auto",  # auto | float16 | int8 | ...
    "language": "auto",
    "dictation_hotkey": "ctrl+alt+d",
    "meeting_hotkey": "ctrl+alt+m",
    "restore_clipboard": False,
    "meetings_dir": "meetings",
    "history_file": "history/dictations.jsonl",
    "custom_words": [],
    "beeps": True,
    "mic_device": None,
    "system_audio_device": None,
    "diarize": True,
    "diarize_threshold": 0.6,
    "diarize_max_speakers": 8,
    "dedup_echo": True,
}

SAMPLE_RATE = 16000  # what whisper wants; mic is captured at this rate when possible


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    else:
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return cfg


def play_tones(spec):
    """Play one tone (freq, ms) or a sequence [(freq, ms), ...] on the default
    output. Cross-platform replacement for winsound.Beep; failures are ignored."""
    if isinstance(spec, tuple):
        spec = [spec]
    try:
        rate = 44100
        parts = []
        for freq, ms in spec:
            t = np.arange(int(rate * ms / 1000)) / rate
            tone = 0.25 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            tone[:200] *= np.linspace(0, 1, 200)   # declick
            tone[-200:] *= np.linspace(1, 0, 200)
            parts.append(tone)
        sd.play(np.concatenate(parts), rate)
    except Exception:
        pass


def beep(cfg, spec):
    if cfg["beeps"]:
        threading.Thread(target=play_tones, args=(spec,), daemon=True).start()


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def list_input_devices():
    """[(index, name, hostapi_name)] for all devices with input channels."""
    apis = sd.query_hostapis()
    return [
        (i, d["name"], apis[d["hostapi"]]["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


# Windows exposes each mic through several host APIs; reliability varies a lot.
_API_RANK = {"windows wasapi": 0, "windows wdm-ks": 1, "windows directsound": 2, "mme": 3}


def resolve_mic_candidates(cfg):
    """Ordered [(index, name)] of devices to try for the mic.

    cfg['mic_device'] may be an integer index or a case-insensitive name
    substring (e.g. "jabra"); all matching device instances are returned,
    most-reliable host API first. With no config, the OS default input is
    tried first, then other instances of the same endpoint.
    """
    want = cfg.get("mic_device")
    inputs = list_input_devices()

    def rank(api):
        return _API_RANK.get(api.lower(), 0)  # CoreAudio/ALSA etc. rank first

    if want is not None:
        if isinstance(want, int):
            if any(i == want for i, *_ in inputs):
                return [(want, sd.query_devices(want)["name"])]
            raise RuntimeError(f"mic_device index {want} has no input channels")
        matches = [t for t in inputs if str(want).lower() in t[1].lower()]
        if not matches:
            raise RuntimeError(f"no input device matching '{want}'")
        matches.sort(key=lambda t: rank(t[2]))
        return [(i, n) for i, n, _a in matches]

    default = sd.default.device[0]
    if default < 0:
        lines = "\n".join(f"    [{i}] {name}  ({api})" for i, name, api in inputs)
        raise RuntimeError(
            "no default microphone reported by the OS (common over RDP without "
            "mic redirection).\n  Available input devices:\n" + (lines or "    (none)")
            + '\n  Fix: set "mic_device" in config.json to an index or name '
            "substring above."
        )
    name = sd.query_devices(default)["name"]
    # other host-API instances of the same endpoint (MME truncates names,
    # so match on prefix in both directions)
    same = [t for t in inputs
            if t[0] != default and (t[1].startswith(name) or name.startswith(t[1]))]
    same.sort(key=lambda t: rank(t[2]))
    return [(default, name)] + [(i, n) for i, n, _a in same]


def open_mic_stream(cfg, make_callback):
    """Open the configured mic, trying every matching device instance and,
    per device, 16 kHz mono first then the native rate/channels (capture
    cards etc. often refuse 16k mono).

    make_callback(rate, channels) must return a sounddevice callback.
    Returns (stream, rate, channels).
    """
    candidates = resolve_mic_candidates(cfg)
    errors = []
    for dev, name in candidates:
        info = sd.query_devices(dev)
        native_rate = int(info["default_samplerate"])
        native_ch = max(1, min(2, info["max_input_channels"]))
        # reported channel counts are often wrong (e.g. mono speakerphones
        # listed as stereo), so also try mono at the native rate
        attempts = [(SAMPLE_RATE, 1), (native_rate, native_ch), (native_rate, 1)]
        attempts = list(dict.fromkeys(attempts))
        for rate, ch in attempts:
            try:
                stream = sd.InputStream(
                    device=dev, samplerate=rate, channels=ch,
                    dtype="float32", callback=make_callback(rate, ch),
                )
                stream.start()
                if len(errors) > 0 or (rate, ch) != (SAMPLE_RATE, 1):
                    log(f"Mic: [{dev}] {name} @ {rate} Hz/{ch}ch")
                return stream, rate, ch
            except Exception as e:
                errors.append(f"[{dev}] {name} @ {rate}Hz/{ch}ch: {e}")
    detail = "\n    ".join(errors[-8:])
    raise RuntimeError(
        "could not open any matching microphone:\n    " + detail +
        "\n  If this is a line/mic jack, check that something is plugged in; "
        "otherwise set \"mic_device\" to another device from --list."
    )


def to_whisper_audio(audio, rate):
    """Downmix to mono and resample to 16 kHz for in-memory transcription."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def fmt_clock(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Keyboard: hotkeys, modifier tracking, paste (pynput, cross-platform)
# ---------------------------------------------------------------------------
def to_pynput_combo(combo):
    """'ctrl+alt+d' -> '<ctrl>+<alt>+d' (pynput GlobalHotKeys syntax)."""
    parts = []
    for tok in combo.lower().split("+"):
        tok = tok.strip()
        tok = {"win": "cmd", "windows": "cmd", "super": "cmd"}.get(tok, tok)
        parts.append(f"<{tok}>" if len(tok) > 1 else tok)
    return "+".join(parts)


class ModifierTracker:
    """Tracks currently held modifier keys so paste can wait for release."""

    def __init__(self):
        self._down = set()
        names = (
            "ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r", "alt_gr",
            "shift", "shift_l", "shift_r", "cmd", "cmd_l", "cmd_r",
        )
        self._mods = {getattr(pk.Key, n) for n in names if hasattr(pk.Key, n)}
        pk.Listener(on_press=self._press, on_release=self._release, daemon=True).start()

    def _press(self, key):
        if key in self._mods:
            self._down.add(key)

    def _release(self, key):
        self._down.discard(key)

    def any_pressed(self):
        return bool(self._down)


def send_paste():
    """Send the platform paste chord to the focused window."""
    kb = pk.Controller()
    mod = pk.Key.cmd if IS_MAC else pk.Key.ctrl
    kb.press(mod)
    kb.press("v")
    kb.release("v")
    kb.release(mod)


# ---------------------------------------------------------------------------
# Transcription engine (shared by dictation and meeting mode)
# ---------------------------------------------------------------------------
def pick_backend(cfg):
    """Resolve (backend, device, compute_type) from config.

    auto rules: NVIDIA GPU -> faster-whisper on CUDA float16. Otherwise use
    whisper-cpp if installed (its builds cover Apple Metal and AMD/Intel
    Vulkan GPUs), else faster-whisper on CPU int8.
    """
    backend, device, compute = cfg["backend"], cfg["device"], cfg["compute_type"]

    has_cuda = False
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass

    if backend == "auto":
        if has_cuda and device in ("auto", "cuda"):
            backend = "faster-whisper"
        else:
            try:
                import pywhispercpp  # noqa: F401
                backend = "whisper-cpp"
            except ImportError:
                backend = "faster-whisper"

    if backend == "faster-whisper":
        if device == "auto":
            device = "cuda" if has_cuda else "cpu"
        if device == "cuda" and not has_cuda:
            log("!! No CUDA device found; falling back to CPU")
            device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
    return backend, device, compute


class Engine:
    """faster-whisper (CUDA/CPU) or whisper.cpp (Metal/Vulkan/CPU), behind one
    interface: transcribe() -> [(start, end, text)]."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.language = None if cfg["language"] == "auto" else cfg["language"]
        self.initial_prompt = (
            ", ".join(cfg["custom_words"]) if cfg["custom_words"] else None
        )
        self.backend, device, compute = pick_backend(cfg)
        t0 = time.time()
        if self.backend == "whisper-cpp":
            log(f"Loading model '{cfg['model']}' via whisper.cpp ...")
            from pywhispercpp.model import Model as CppModel
            self.model = CppModel(cfg["model"])
        else:
            log(f"Loading model '{cfg['model']}' via faster-whisper on {device} ({compute}) ...")
            self.model = WhisperModel(cfg["model"], device=device, compute_type=compute)
        log(f"Model ready in {time.time() - t0:.1f}s")

    def _load(self, audio_or_path):
        """whisper.cpp wants a 16k mono float32 array; load files ourselves.
        (faster-whisper decodes paths itself via PyAV, video containers
        included, so this path is only exercised by the whisper-cpp backend.)
        """
        if not isinstance(audio_or_path, str):
            return audio_or_path
        try:
            import soundfile as sf
            data, rate = sf.read(audio_or_path, dtype="float32")
            return to_whisper_audio(data, rate)
        except Exception:
            # soundfile only reads plain audio containers (wav/flac/...);
            # fall back to PyAV for video files (mp4, mkv, ...).
            return self._decode_any_media(audio_or_path)

    @staticmethod
    def _decode_any_media(path):
        import av

        container = av.open(path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                arr = rframe.to_ndarray().reshape(-1)
                if arr.size:
                    chunks.append(arr)
        container.close()
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def transcribe(self, audio_or_path, task="transcribe"):
        """Returns a list of (start, end, text) segments.

        task: "transcribe" (source language) or "translate" (Whisper's
        built-in translate-to-English task).
        """
        with self.lock:
            if self.backend == "whisper-cpp":
                kwargs = {"language": self.language or "auto"}
                if self.initial_prompt:
                    kwargs["initial_prompt"] = self.initial_prompt
                if task == "translate":
                    kwargs["translate"] = True
                segs = self.model.transcribe(self._load(audio_or_path), **kwargs)
                return [
                    (s.t0 / 100.0, s.t1 / 100.0, s.text.strip())
                    for s in segs if s.text.strip()
                ]
            segments, _info = self.model.transcribe(
                audio_or_path,
                task=task,
                language=self.language,
                vad_filter=True,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=False,  # avoids hallucination loops on silence
            )
            return [(s.start, s.end, s.text.strip()) for s in segments]


# ---------------------------------------------------------------------------
# Dictation: toggle mic recording, transcribe, paste into focused window
# ---------------------------------------------------------------------------
class Dictation:
    def __init__(self, cfg, engine, mods):
        self.cfg = cfg
        self.engine = engine
        self.mods = mods
        self.recording = False
        self.chunks = []
        self.stream = None
        self.rate = SAMPLE_RATE
        self.history_path = BASE_DIR / cfg["history_file"]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def toggle(self):
        if self.recording:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.chunks = []
        try:
            self.stream, self.rate, _ch = open_mic_stream(
                self.cfg,
                lambda rate, ch: lambda indata, *_: self.chunks.append(indata.copy()),
            )
        except Exception as e:
            log(f"!! Could not open microphone: {e}")
            return
        self.recording = True
        beep(self.cfg, (880, 120))
        log("Dictation: recording... (press hotkey again to stop)")

    def _stop(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        beep(self.cfg, (660, 120))
        if not self.chunks:
            log("Dictation: no audio captured")
            return
        audio = to_whisper_audio(np.concatenate(self.chunks), self.rate)
        dur = len(audio) / SAMPLE_RATE
        log(f"Dictation: transcribing {dur:.1f}s of audio ...")
        threading.Thread(target=self._finish, args=(audio,), daemon=True).start()

    def _finish(self, audio):
        t0 = time.time()
        try:
            segments = self.engine.transcribe(audio)
        except Exception as e:
            log(f"!! Transcription failed: {e}")
            return
        text = " ".join(s[2] for s in segments).strip()
        if not text:
            log("Dictation: (nothing recognized)")
            return
        log(f'Dictation ({time.time() - t0:.1f}s): "{text}"')
        self._paste(text)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"time": datetime.now().isoformat(timespec="seconds"), "text": text},
                ensure_ascii=False) + "\n")

    def _paste(self, text):
        old = None
        if self.cfg["restore_clipboard"]:
            try:
                old = pyperclip.paste()
            except Exception:
                pass
        try:
            pyperclip.copy(text)
        except Exception as e:
            hint = " (install xclip or xsel)" if sys.platform.startswith("linux") else ""
            log(f"!! Clipboard unavailable{hint}: {e}")
            return
        # wait until hotkey modifiers are released so the paste chord isn't mangled
        deadline = time.time() + 3
        while time.time() < deadline and self.mods.any_pressed():
            time.sleep(0.05)
        send_paste()
        beep(self.cfg, (523, 100))
        if old is not None:
            time.sleep(0.5)
            try:
                pyperclip.copy(old)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Speaker diarization for the remote side of a meeting (fully local).
# ECAPA voice embeddings per transcript segment + agglomerative clustering.
# ---------------------------------------------------------------------------
class Diarizer:
    MIN_SEG_SECONDS = 0.7  # too little voice -> unreliable embedding

    def __init__(self, cfg):
        from speechbrain.inference.speaker import EncoderClassifier

        log("Loading speaker-embedding model (first run downloads ~80MB) ...")
        # COPY_SKIP_CACHE writes real files; the default SYMLINK strategy
        # fails on Windows without Developer Mode
        from speechbrain.utils.fetching import LocalStrategy

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(BASE_DIR / "models" / "ecapa"),
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        self.threshold = cfg["diarize_threshold"]
        self.max_speakers = cfg.get("diarize_max_speakers", 8)

    def _cluster(self, X):
        """Cluster embeddings into speakers via spectral clustering with
        eigengap-based speaker counting (the standard diarization recipe).
        Agglomerative/threshold approaches degenerate on real meeting audio:
        codec artifacts chain everything into one blob plus outlier crumbs.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.cluster.vq import kmeans2
        from scipy.spatial.distance import pdist

        n = len(X)
        if n == 1:
            return np.array([1])
        if n <= 3:  # too few points for a similarity graph
            return fcluster(linkage(pdist(X, metric="cosine"), method="average"),
                            t=self.threshold, criterion="distance")

        # sparsified cosine-affinity graph: keep each row's strongest links
        sim = X @ X.T
        np.fill_diagonal(sim, 0)
        keep = max(2, int(0.06 * n))
        A = np.zeros_like(sim)
        for i in range(n):
            top = np.argpartition(sim[i], -keep)[-keep:]
            A[i, top] = sim[i, top]
        A = np.maximum(A, A.T)

        # normalized Laplacian; number of speakers = position of the largest
        # gap in the smallest eigenvalues
        deg = np.maximum(A.sum(axis=1), 1e-9)
        dinv = 1.0 / np.sqrt(deg)
        L = np.eye(n) - (dinv[:, None] * A) * dinv[None, :]
        evals, evecs = np.linalg.eigh(L)
        kmax = min(self.max_speakers, n - 1)
        k = int(np.argmax(np.diff(evals[: kmax + 1])) + 1)
        if k == 1:
            return np.ones(n, dtype=int)

        # k-means on the first k (row-normalized) eigenvectors
        V = evecs[:, :k]
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        best_labels, best_inertia = None, None
        for seed in range(3):
            centroids, labels = kmeans2(V, k, minit="++", seed=seed)
            inertia = float(((V - centroids[labels]) ** 2).sum())
            if best_inertia is None or inertia < best_inertia:
                best_labels, best_inertia = labels, inertia
        labels = best_labels + 1

        # fold clusters with too little speech into the nearest big cluster
        D = 1 - sim
        min_size = max(2, int(0.02 * n))
        big = [c for c in np.unique(labels) if (labels == c).sum() >= min_size]
        if big:
            for c in np.unique(labels):
                if c in big:
                    continue
                idx = np.where(labels == c)[0]
                dists = [(D[np.ix_(idx, np.where(labels == b)[0])].mean(), b)
                         for b in big]
                labels[idx] = min(dists)[1]
        return labels

    def label(self, wav_path, segments):
        """Assign a speaker number to each (start, end, text) segment.

        Returns a list of ints (1-based, in order of first appearance).
        Segments too short to embed inherit the previous segment's speaker.
        """
        import torch
        import soundfile as sf

        data, rate = sf.read(wav_path, dtype="float32")
        audio = to_whisper_audio(data, rate)

        embeddings, embedded_idx = [], []
        for k, (start, end, _text) in enumerate(segments):
            # whisper timestamps can overshoot the file; clamp before slicing
            s = min(int(start * SAMPLE_RATE), len(audio))
            e = min(int(end * SAMPLE_RATE), len(audio))
            if (e - s) / SAMPLE_RATE < self.MIN_SEG_SECONDS:
                continue
            chunk = torch.from_numpy(audio[s:e]).unsqueeze(0)
            try:
                with torch.no_grad():
                    emb = self.encoder.encode_batch(chunk).squeeze().cpu().numpy()
            except Exception:
                continue  # bad chunk -> segment inherits a neighbor's speaker
            embeddings.append(emb / (np.linalg.norm(emb) + 1e-9))
            embedded_idx.append(k)

        if len(embeddings) == 0:
            return [1] * len(segments)
        clusters = self._cluster(np.array(embeddings))

        # renumber clusters by first appearance, then fill short segments
        order, speakers = {}, {}
        for idx, c in zip(embedded_idx, clusters):
            order.setdefault(c, len(order) + 1)
            speakers[idx] = order[c]
        labels, prev = [], 1
        for k in range(len(segments)):
            prev = speakers.get(k, prev)
            labels.append(prev)
        return labels


# ---------------------------------------------------------------------------
# Meeting recording: mic + system audio -> merged, speaker-labeled transcript
# ---------------------------------------------------------------------------
class Meeting:
    def __init__(self, cfg, engine):
        self.cfg = cfg
        self.engine = engine
        self.recording = False
        self.meetings_dir = BASE_DIR / cfg["meetings_dir"]
        self.meetings_dir.mkdir(parents=True, exist_ok=True)

    def toggle(self):
        if self.recording:
            self._stop()
        else:
            self._start()

    # -- system audio (what you hear), per platform -------------------------
    def _open_system_capture(self, out_path):
        """Start capturing system playback audio to out_path.
        Sets self._sys_close to a cleanup callable. Returns the source name."""
        override = self.cfg.get("system_audio_device")

        if IS_WIN:
            import pyaudiowpatch as pyaudio

            pa = pyaudio.PyAudio()
            speakers = None
            if override is not None:
                for lb in pa.get_loopback_device_info_generator():
                    if str(override).lower() in lb["name"].lower():
                        speakers = lb
                        break
                if speakers is None:
                    pa.terminate()
                    raise RuntimeError(f"no loopback device matching '{override}'")
            else:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                if wasapi["defaultOutputDevice"] < 0:
                    pa.terminate()
                    raise RuntimeError("no default output device (check sound settings)")
                speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
                if not speakers.get("isLoopbackDevice"):
                    for lb in pa.get_loopback_device_info_generator():
                        if speakers["name"] in lb["name"]:
                            speakers = lb
                            break
                    else:
                        pa.terminate()
                        raise RuntimeError("no loopback device for default output")
            rate = int(speakers["defaultSampleRate"])
            channels = max(1, int(speakers["maxInputChannels"]))

            wf = wave.open(str(out_path), "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(rate)

            def cb(in_data, frame_count, time_info, status):
                wf.writeframes(in_data)
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
                wf.close()

            self._sys_close = close
            return speakers["name"]

        # macOS / Linux: capture from a monitor or virtual-loopback input
        inputs = list_input_devices()
        dev = None
        if override is not None:
            for i, name, _api in inputs:
                if str(override).lower() in name.lower():
                    dev = i
                    break
            if dev is None:
                raise RuntimeError(f"no input device matching '{override}'")
        else:
            keys = ("blackhole", "loopback", "soundflower") if IS_MAC else ("monitor",)
            for i, name, _api in inputs:
                if any(k in name.lower() for k in keys):
                    dev = i
                    break
            if dev is None:
                if IS_MAC:
                    raise RuntimeError(
                        "no virtual loopback input found. Install BlackHole "
                        "(https://existential.audio/blackhole/), create a "
                        "Multi-Output Device (speakers + BlackHole) in Audio "
                        "MIDI Setup, and use it as the system output during "
                        "meetings — or set \"system_audio_device\" in config.json"
                    )
                raise RuntimeError(
                    "no PulseAudio/PipeWire monitor input found — set "
                    "\"system_audio_device\" in config.json (see --list)"
                )
        info = sd.query_devices(dev)
        rate = int(info["default_samplerate"])
        channels = max(1, min(2, info["max_input_channels"]))

        wf = wave.open(str(out_path), "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)

        def cb(indata, *_):
            wf.writeframes((np.clip(indata, -1, 1) * 32767).astype(np.int16).tobytes())

        stream = sd.InputStream(
            device=dev, samplerate=rate, channels=channels,
            dtype="float32", callback=cb,
        )
        stream.start()

        def close():
            stream.stop()
            stream.close()
            wf.close()

        self._sys_close = close
        return info["name"]

    def _start(self):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = self.meetings_dir / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()

        # mic -> wav (int16, at whatever format the device supports)
        mic_wav_path = self.dir / "mic.wav"
        self._mic_wave = None

        def make_cb(rate, ch):
            if self._mic_wave is not None:  # earlier format attempt failed
                self._mic_wave.close()
            wf = wave.open(str(mic_wav_path), "wb")
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            self._mic_wave = wf

            def cb(indata, *_):
                wf.writeframes(
                    (np.clip(indata, -1, 1) * 32767).astype(np.int16).tobytes()
                )
            return cb

        try:
            self._mic_stream, _rate, _ch = open_mic_stream(self.cfg, make_cb)
        except Exception as e:
            log(f"!! Could not open microphone: {e}")
            return

        self._sys_close = None
        try:
            source = self._open_system_capture(self.dir / "system.wav")
            log(f"Meeting: capturing system audio from '{source}'")
        except Exception as e:
            log(f"!! System-audio capture unavailable ({e}); recording mic only")

        self.recording = True
        beep(self.cfg, (880, 250))
        log(f"Meeting: RECORDING -> {self.dir}  (press hotkey again to stop)")

    def _stop(self):
        self.recording = False
        dur = time.time() - self.t_start
        self._mic_stream.stop()
        self._mic_stream.close()
        self._mic_wave.close()
        if self._sys_close is not None:
            self._sys_close()
        beep(self.cfg, (660, 200))
        log(f"Meeting: stopped after {fmt_clock(dur)}. Transcribing in background ...")
        threading.Thread(target=self._finish, args=(dur,), daemon=True).start()

    @staticmethod
    def _drop_echo(mine, remote):
        """Remove mic segments that are just the speakerphone playing the far
        end (echo). A mic segment whose words mostly appear in the remote
        transcript around the same time is bleed, not the user talking."""
        import re

        kept = []
        for seg in mine:
            start, end, text = seg
            window = " ".join(
                r[2] for r in remote if r[0] < end + 4 and r[1] > start - 4
            ).lower()
            wtok = set(re.findall(r"[a-z0-9']+", window))
            stok = re.findall(r"[a-z0-9']+", text.lower())
            if stok and wtok:
                overlap = sum(t in wtok for t in stok) / len(stok)
                if overlap >= 0.6:
                    continue
            kept.append(seg)
        return kept

    def _get_diarizer(self):
        if not self.cfg["diarize"]:
            return None
        if not hasattr(self, "_diarizer"):
            try:
                self._diarizer = Diarizer(self.cfg)
            except Exception as e:
                log(f"!! Diarization unavailable ({e}); labeling all remote speech 'Them'")
                self._diarizer = None
        return self._diarizer

    def _finish(self, dur):
        t0 = time.time()
        entries = []  # (start_seconds, speaker, text)
        try:
            mine = self.engine.transcribe(str(self.dir / "mic.wav"))
            remote = []
            sys_wav = self.dir / "system.wav"
            if sys_wav.exists() and sys_wav.stat().st_size > 1024:
                remote = self.engine.transcribe(str(sys_wav))
            if remote and self.cfg.get("dedup_echo", True):
                before = len(mine)
                mine = self._drop_echo(mine, remote)
                if len(mine) < before:
                    log(f"Meeting: dropped {before - len(mine)} echo segments "
                        "(far-end audio picked up by the mic)")
            for s in mine:
                entries.append((s[0], "Me", s[2]))
            if remote:
                labels = [1] * len(remote)
                diarizer = self._get_diarizer() if remote else None
                if diarizer is not None:
                    try:
                        labels = diarizer.label(str(sys_wav), remote)
                    except Exception as e:
                        log(f"!! Diarization failed ({e}); labeling all remote speech 'Them'")
                n_speakers = max(labels, default=1)
                for s, n in zip(remote, labels):
                    who = "Them" if n_speakers == 1 else f"Them-{n}"
                    entries.append((s[0], who, s[2]))
                if n_speakers > 1:
                    log(f"Meeting: {n_speakers} distinct remote speakers detected")
        except Exception as e:
            log(f"!! Meeting transcription failed: {e}")
            return
        entries.sort(key=lambda e: e[0])

        out = self.dir / "transcript.md"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# Meeting {self.dir.name}\n\n")
            f.write(f"Duration: {fmt_clock(dur)}\n\n")
            for start, speaker, text in entries:
                f.write(f"**[{fmt_clock(start)}] {speaker}:** {text}\n\n")
        log(f"Meeting: transcript ready ({time.time() - t0:.1f}s) -> {out}")
        beep(self.cfg, [(523, 100), (659, 100)])


# ---------------------------------------------------------------------------
def print_devices():
    """List every audio input (mics, capture cards) and system-audio source."""
    print("\nAudio inputs (use index or a name fragment as \"mic_device\" in config.json):")
    inputs = list_input_devices()
    if not inputs:
        print("  (none found)")
    for i, name, api in inputs:
        print(f"  [{i}] {name}  ({api})")
    default_in = sd.default.device[0]
    if default_in >= 0:
        print(f"\nDefault input: [{default_in}] {sd.query_devices(default_in)['name']}")
    else:
        print("\nDefault input: none reported by the OS")
    print("\nSystem-audio sources (used by meeting mode):")
    if IS_WIN:
        try:
            import pyaudiowpatch as pyaudio

            p = pyaudio.PyAudio()
            names = [lb["name"] for lb in p.get_loopback_device_info_generator()]
            p.terminate()
            for name in names or ["(none found)"]:
                print(f"  {name}")
        except Exception as e:
            print(f"  loopback enumeration failed: {e}")
    else:
        keys = ("blackhole", "loopback", "soundflower") if IS_MAC else ("monitor",)
        found = [n for _i, n, _a in inputs if any(k in n.lower() for k in keys)]
        for name in found:
            print(f"  {name}")
        if not found:
            if IS_MAC:
                print("  (none — install BlackHole for system-audio capture)")
            else:
                print("  (none — is PulseAudio/PipeWire running?)")
    print()


def main():
    if "--list" in sys.argv:
        print_devices()
        return

    if pk is None:
        print(f"Global hotkeys unavailable: {PYNPUT_ERROR}\n"
              "On Linux this usually means no X11/desktop session "
              "(Wayland users: enable XWayland).", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    engine = Engine(cfg)
    mods = ModifierTracker()
    dictation = Dictation(cfg, engine, mods)
    meeting = Meeting(cfg, engine)

    try:
        candidates = resolve_mic_candidates(cfg)
        extra = f" (+{len(candidates) - 1} fallback instances)" if len(candidates) > 1 else ""
        log(f"Microphone: {candidates[0][1]}{extra}")
    except RuntimeError as e:
        log(f"!! {e}")

    hotkeys = pk.GlobalHotKeys({
        to_pynput_combo(cfg["dictation_hotkey"]): dictation.toggle,
        to_pynput_combo(cfg["meeting_hotkey"]): meeting.toggle,
    })
    hotkeys.daemon = True
    hotkeys.start()

    print()
    log(f"Ready.  {cfg['dictation_hotkey']} = dictate & paste   |   "
        f"{cfg['meeting_hotkey']} = meeting record   |   Ctrl+C here = quit")
    print()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        if dictation.recording:
            dictation.stream.stop()
        if meeting.recording:
            meeting._stop()
    log("Bye.")


if __name__ == "__main__":
    main()
