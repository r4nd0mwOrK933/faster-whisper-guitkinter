"""
Runtime hook for Faster-Whisper frozen application.
Executed by PyInstaller bootloader before the main script (whisper_tkgui.py).

This hook ensures correct module resolution in the hybrid layout:
- internal/  : compiled third-party dependencies (PyInstaller managed)
- app/       : editable source package (in dist root, not compiled)
- *.py       : editable top-level scripts (whisper_tkgui.py, app_gui.py)
"""

import os
import sys
from pathlib import Path


def _is_frozen():
    """Check if running in a PyInstaller-frozen environment."""
    return getattr(sys, 'frozen', False)


def _get_app_dir():
    """Get the application root directory (where the .exe lives)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    else:
        # When running from source (python whisper_tkgui.py)
        return Path(__file__).resolve().parent


def _setup_sys_path():
    """
    Ensure sys.path includes:
    1. internal/  — PyInstaller's contents_directory with compiled deps
    2. App root   — for editable app/ package and top-level .py files
    """
    app_dir = _get_app_dir()

    # internal/ subdirectory (PyInstaller contents_directory='internal')
    internal_dir = app_dir / 'internal'
    if internal_dir.is_dir():
        internal_str = str(internal_dir)
        if internal_str not in sys.path:
            sys.path.insert(0, internal_str)

    # App root directory (for editable sources)
    app_dir_str = str(app_dir)
    if app_dir_str not in sys.path:
        sys.path.insert(0, app_dir_str)


def _patch_faster_whisper_assets():
    """
    Ensure faster_whisper.utils.get_assets_path() returns the correct path
    when running in frozen mode.

    The Silero VAD model (silero_vad_v6.onnx) is collected as a data file
    into internal/faster_whisper/assets/. The default __file__-based
    resolution should work, but we add a safety patch here.
    """
    if not _is_frozen():
        return

    try:
        import faster_whisper.utils as fw_utils

        # Store original for reference
        _original_get_assets_path = fw_utils.get_assets_path

        def _patched_get_assets_path():
            """Resolve assets path correctly in frozen mode."""
            # faster_whisper.utils lives in internal/faster_whisper/utils.pyc
            # The assets directory is alongside it: internal/faster_whisper/assets/
            utils_file = Path(fw_utils.__file__)
            assets_dir = utils_file.parent / 'assets'
            if assets_dir.is_dir():
                return str(assets_dir)
            # Fallback: search internal/ for faster_whisper/assets
            app_dir = _get_app_dir()
            candidate = app_dir / 'internal' / 'faster_whisper' / 'assets'
            if candidate.is_dir():
                return str(candidate)
            # Last resort: use original implementation
            return _original_get_assets_path()

        fw_utils.get_assets_path = _patched_get_assets_path

    except ImportError:
        # faster_whisper not yet importable — will be patched when loaded
        pass


# ---- Execute on import ----
_setup_sys_path()
_patch_faster_whisper_assets()
