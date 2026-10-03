# Faster Whisper GUI

A Windows desktop GUI for local speech-to-text transcription and subtitle generation, built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features

- Chinese and English Tkinter interfaces
- FireRedVAD voice activity detection before transcription
- Local CTranslate2 Whisper model support
- SRT subtitle generation
- CPU and Windows DirectML execution paths
- PyInstaller build configuration for a portable Windows folder

## Requirements

- Windows 10 or later
- Python 3.14
- FFmpeg available on `PATH`
- A locally downloaded CTranslate2-format Whisper model

The application calls the `ffmpeg` command-line program for audio decoding. FFmpeg is not bundled by this project; install a redistributable FFmpeg build separately and follow its license terms.

## Install and run from source

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
# Debug launch: keep the terminal visible to inspect error output
python whisper_tkgui.py
```

For normal source-based use, you can use `pythonw.exe` to hide the console window:

```powershell
pythonw.exe whisper_tkgui.py
```

The packaged Windows GUI application also uses windowed mode without a console. Run
`dist\whisper_tkgui\whisper_tkgui.exe` directly.

The first run generates `gui_config.ini` from gui_config.example.ini. Fill in the local paths for the Whisper model and audio file. Do not commit personal paths or credentials.

The GUI expects a local CTranslate2 model directory. It does not download the main Whisper model automatically. Models converted or downloaded from the [faster-whisper model collection](https://huggingface.co/Systran) can be selected by their local directory path.

FireRedVAD and Silero VAD runtime resources are included in this repository. The main Whisper model is intentionally not included because model files are large and have their own distribution terms.

## Build the Windows application

Run these commands from the repository root in a clean Python environment:

```powershell
python -m pip install ".[build]"
python -m PyInstaller --clean --noconfirm build.spec
```

The output is created at `dist/whisper_tkgui/`. The directory is a folder-based application and should be distributed as a ZIP archive. The output includes `LICENSE`, `THIRD_PARTY_NOTICES`, and the related files under `licenses/`. FFmpeg remains a separate Windows prerequisite.

## GitHub Actions

- `Windows GUI Build` can be started manually from the Actions page.
- The generated Windows ZIP is uploaded as a workflow artifact for download.
- This workflow does not run automatically on pushes, pull requests, or tags, and does not create GitHub Releases.

The workflow never downloads or packages the main Whisper model.

## License and attribution

This project is a derivative application that includes code from faster-whisper. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for third-party licenses, model sources, checksums, and attribution requirements.
