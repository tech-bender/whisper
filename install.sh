#!/usr/bin/env bash
# WhisperFlow-local installer (macOS / Linux)
# Run from this folder:  bash install.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v python3 >/dev/null; then
    echo "Python 3.10+ is required."; exit 1
fi

echo "Installing WhisperFlow-local ..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install "$DIR"

case "$(uname)" in
  Darwin)
    # whisper.cpp backend gives Apple-silicon GPU (Metal) acceleration
    echo "macOS: installing whisper.cpp backend for Metal GPU support ..."
    python3 -m pip install "$DIR[whispercpp]" || echo "whisper.cpp backend optional - continuing"
    echo ""
    echo "macOS notes:"
    echo " - Grant Accessibility + Input Monitoring permission to your terminal"
    echo "   (System Settings > Privacy & Security) for global hotkeys/paste."
    echo " - For meeting system-audio capture install BlackHole:"
    echo "   https://existential.audio/blackhole/  then create a Multi-Output"
    echo "   Device in Audio MIDI Setup and select it as output during meetings."
    ;;
  Linux)
    if command -v nvidia-smi >/dev/null; then
        echo "NVIDIA GPU detected - installing CUDA libraries ..."
        python3 -m pip install "$DIR[cuda]"
    else
        echo "No NVIDIA GPU detected - CPU mode (or install a Vulkan pywhispercpp build for AMD/Intel GPU)."
    fi
    echo ""
    echo "Linux notes:"
    echo " - Clipboard paste needs xclip:  sudo apt install xclip  (or xsel)"
    echo " - Global hotkeys need X11/XWayland (pure Wayland is not supported)."
    echo " - Meeting mode records the PulseAudio/PipeWire 'monitor' source automatically."
    ;;
esac

echo ""
echo "Done. Start with:  whisperflow"
echo "List audio devices:  whisperflow --list"
echo "Config: ~/.whisperflow/config.json (created on first run)"
