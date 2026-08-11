@echo off
title Audio devices
cd /d "%~dp0"
python whisperflow.py --list
pause
