"""
subtitles.py: turn a local video (or audio) file into an .srt subtitle file,
optionally translated to English -- fully on-device, for playback in VLC.

    python subtitles.py movie.mkv                  # subtitles in the source language
    python subtitles.py movie.mkv --translate       # English subtitles, any source language
    python subtitles.py movie.mkv --language ja     # skip language auto-detect
    python subtitles.py movie.mkv -o movie.en.srt   # custom output path

By default the .srt is written next to the video with the same base name
(movie.mkv -> movie.srt). VLC auto-loads a subtitle file matching that
convention as soon as you open the video -- no extra setup. Otherwise use
VLC's Subtitle -> Add Subtitle File... and point it at the output.

This is a batch step, not a live captioner: Whisper needs a full pass over
the audio (or file) to keep accuracy and timestamps decent, so run it before
pressing play. For a file that's minutes to a couple hours long this still
finishes well ahead of watch time on a GPU. See README.md for a true-live
overlay if that's ever needed instead.

Video containers (mp4, mkv, ...) are decoded directly -- no ffmpeg binary
required, faster-whisper/PyAV read the audio track on their own.
"""
import argparse
import sys
from pathlib import Path

from whisperflow import Engine, load_config, log


def fmt_srt_time(seconds):
    ms = round(max(seconds, 0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, 1):
            f.write(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{text}\n\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate an .srt subtitle file (optionally translated to "
        "English) from a local video/audio file, for VLC playback.",
    )
    ap.add_argument("video", help="path to a video or audio file")
    ap.add_argument(
        "-o", "--output",
        help="output .srt path (default: <video>.srt next to the video, "
        "so VLC auto-loads it)",
    )
    ap.add_argument(
        "--translate", action="store_true",
        help="translate speech to English subtitles (Whisper's built-in "
        "translate task; English is the only translation target it supports)",
    )
    ap.add_argument(
        "--language",
        help="force the source language (e.g. 'en', 'ja'); default is "
        "config.json's setting, or auto-detect",
    )
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"error: file not found: {video_path}")
    out_path = Path(args.output) if args.output else video_path.with_suffix(".srt")

    cfg = load_config()
    if args.language:
        cfg["language"] = args.language

    engine = Engine(cfg)
    task = "translate" if args.translate else "transcribe"
    log(f"Transcribing '{video_path.name}' (task={task}) ...")
    segments = engine.transcribe(str(video_path), task=task)
    if not segments:
        sys.exit("No speech detected.")

    write_srt(segments, out_path)
    log(f"Wrote {len(segments)} subtitle lines -> {out_path}")
    log("Open the video in VLC -- it auto-loads a same-name .srt in the same "
        "folder, or use Subtitle -> Add Subtitle File... to pick it manually.")


if __name__ == "__main__":
    main()
