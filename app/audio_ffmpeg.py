import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np


def _prepare_input_path(input_file: Union[str, os.PathLike, BinaryIO]) -> Tuple[str, str]:
    if isinstance(input_file, (str, os.PathLike)):
        return os.fspath(input_file), ""

    if hasattr(input_file, "seek"):
        input_file.seek(0)

    data = input_file.read()
    suffix = Path(getattr(input_file, "name", "input.bin")).suffix or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data)
    temp_file.flush()
    temp_file.close()
    return temp_file.name, temp_file.name


def decode_audio(
    input_file: Union[str, os.PathLike, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decode audio by invoking ffmpeg from the command line."""
    input_path, temp_path = _prepare_input_path(input_file)
    channels = 2 if split_stereo else 1
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sampling_rate),
        "-ac",
        str(channels),
        "-",
    ]

    try:
        process = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as error:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError("ffmpeg executable was not found on PATH") from error

    try:
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to decode audio: {stderr or 'unknown error'}")

        audio = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if split_stereo:
        return audio[0::2], audio[1::2]

    return audio


def pad_or_trim(array: np.ndarray, length: int = 3000, *, axis: int = -1):
    """Pad or trim the Mel features array to the requested length."""
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array
