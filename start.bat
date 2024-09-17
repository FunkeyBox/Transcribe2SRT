@echo off
setlocal enabledelayedexpansion

:: Activate the Conda environment
call conda activate whisperx

:: Run the Python script
python app\transcribe2srt.py %*

pause