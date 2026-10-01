"""Subtitle-chunk transcription built on top of WhisperModel.

Provides:
- WhisperSubtitle: a WhisperModel subclass that transcribes audio chunked by
  pre-existing subtitle timestamps (SRT / ASS).
- get_cudnn: load CUDA DLLs from a directory on Windows.
- save_subtitle: write transcription segments to SRT or ASS files.
"""

import ctypes
import dataclasses
import glob
import logging
import os
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from app.audio_ffmpeg import decode_audio
from app.srt_utils import parse_srt_file, srt_time_to_seconds
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    Segment,
    TranscriptionInfo,
    TranscriptionOptions,
    VadOptions,
    WhisperModel,
    get_suppressed_tokens,
)

# ---------------------------------------------------------------------------
# Task cancellation
# ---------------------------------------------------------------------------

class TaskCancelledError(Exception):
    """转录任务被外部取消时抛出（GUI 取消按钮触发）。"""


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _ass_time_to_seconds(timestamp: str) -> float:
    """Convert an ASS timestamp (H:MM:SS.cc) to seconds."""
    timestamp = timestamp.strip()
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse ASS timestamp: {timestamp}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def get_sub_timestamps(
    sub_path: Union[str, Path],
    sample_rate: Optional[int] = None,
) -> List[Dict[str, int]]:
    """Extract speech-chunk boundaries from a subtitle file.

    Supports .srt (via ``srt_utils.parse_srt_file``) and .ass (Dialogue lines).

    Args:
        sub_path: Path to the subtitle file.
        sample_rate: If provided, timestamps are converted to sample indices
            (``int(seconds * sample_rate)``).

    Returns:
        List of ``{"start": …, "end": …}`` dicts (seconds or samples).
    """
    sub_path = Path(sub_path)
    suffix = sub_path.suffix.lower()

    if suffix == ".srt":
        segments = parse_srt_file(str(sub_path))
        time_chunks: List[Dict[str, int]] = []
        for start_s, end_s, _text in segments:
            if sample_rate:
                time_chunks.append(
                    {"start": int(start_s * sample_rate), "end": int(end_s * sample_rate)}
                )
            else:
                time_chunks.append({"start": start_s, "end": end_s})
        return time_chunks

    if suffix == ".ass":
        with open(sub_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        dialogue_lines = [ln for ln in lines if ln.startswith("Dialogue:")]
        time_chunks = []
        for line in dialogue_lines:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            start_sec = _ass_time_to_seconds(parts[1])
            end_sec = _ass_time_to_seconds(parts[2])
            if sample_rate:
                time_chunks.append(
                    {"start": int(start_sec * sample_rate), "end": int(end_sec * sample_rate)}
                )
            else:
                time_chunks.append({"start": start_sec, "end": end_sec})
        return time_chunks

    raise ValueError(f"Unsupported subtitle format: {suffix}")


def collect_chunks(
    audio: np.ndarray,
    chunks: List[Dict[str, int]],
) -> List[np.ndarray]:
    """Slice *audio* into sub-arrays according to *chunks*."""
    return [audio[chunk["start"] : chunk["end"]] for chunk in chunks] if chunks else []


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    """Format *seconds* as ``HH:MM:SS.mmm`` (or ``MM:SS.mmm``)."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


# ---------------------------------------------------------------------------
# CUDA helper
# ---------------------------------------------------------------------------


import os

_dll_dir_handles = []

def load_cudnn(cudnn_dir: str) -> None:
    # """Register CUDA/cuDNN DLL directory on Windows."""
    # if os.name != "nt" or not cudnn_dir:
    #     return
    # _dll_dir_handles.append(os.add_dll_directory(cudnn_dir))
    """Load CUDA libraries from specified directory."""
    for library in glob.glob(os.path.join(cudnn_dir, "*.dll")):
        ctypes.CDLL(library)


# ---------------------------------------------------------------------------
# Subtitle output
# ---------------------------------------------------------------------------


def save_subtitle(
    segments: Iterable[Segment],
    wav_path: Union[str, Path],
    save_format: str = "srt",
    come_from: str = "whisper",
    tmp_name: Optional[str] = None,
) -> Path:
    """Write transcription *segments* to an SRT or ASS file.

    Args:
        segments: Transcribed segments (produced by the model).
        wav_path: Original audio path (used to derive output directory & stem).
        save_format: ``"srt"`` or ``"ass"``.
        come_from: Tag appended to the output filename.
        tmp_name: Override the output file stem.

    Returns:
        Path to the saved subtitle file.
    """
    wav_path = Path(wav_path)
    file_name = tmp_name if tmp_name else wav_path.stem
    save_path = wav_path.parent / f"{file_name}_{come_from}.{save_format}"

    if save_format == "srt":
        with open(save_path, "w", encoding="utf-8") as fh:
            for idx, seg in enumerate(segments, start=1):
                text = seg.text if hasattr(seg, "text") else ""
                start = format_timestamp(seg.start, True, ",")
                end = format_timestamp(seg.end, True, ",")
                fh.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
    elif save_format == "ass":
        header = (
            "[Script Info]\n"
            "; Script generated by faster-whisper app\n"
            "Title: Default Aegisub file\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "YCbCr Matrix: None\n"
            "\n"
            "[Aegisub Project Garbage]\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,1,1,1,10,10,10,1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(header)
            for seg in segments:
                text = seg.text if hasattr(seg, "text") else ""
                start = format_timestamp(seg.start, True)
                end = format_timestamp(seg.end, True)
                fh.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    else:
        raise ValueError(f"Unsupported subtitle save format: {save_format}")

    return save_path


# ---------------------------------------------------------------------------
# WhisperSubtitle
# ---------------------------------------------------------------------------


class WhisperSubtitle(WhisperModel):
    """WhisperModel variant that transcribes audio chunked by subtitle timestamps.

    Instead of using VAD or fixed-length chunks, this class reads an existing
    subtitle file (SRT / ASS) and treats each subtitle entry as a speech chunk.
    """

    def transcribe_with_subtitle_chunks(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        subtitle_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
        cross_chunk_prompt: bool = False,
        prompt_window_tokens: int = 64,
        reset_prompt_on_gap_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribe *audio* using subtitle entries as speech chunks.

        See ``WhisperModel.transcribe`` for parameter details.
        """
        sampling_rate = self.feature_extractor.sampling_rate

        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelledError("任务已取消")

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        speech_chunks = get_sub_timestamps(subtitle_path, sample_rate=sampling_rate)
        audio_chunks = collect_chunks(audio, speech_chunks)
        duration_after_vad = sum(len(c) for c in audio_chunks) / sampling_rate

        self.logger.info(
            "Subtitle filter removed %s of audio",
            format_timestamp(duration - duration_after_vad),
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "Subtitle chunks kept: %s",
                ", ".join(
                    f"[{format_timestamp(chunk['start'] / sampling_rate)} -> "
                    f"{format_timestamp(chunk['end'] / sampling_rate)}]"
                    for chunk in speech_chunks
                ),
            )

        # --- feature extraction per chunk -----------------------------------
        feature_chunks = [
            self.feature_extractor(chunk, chunk_length=chunk_length) for chunk in audio_chunks
        ]

        # --- language detection ---------------------------------------------
        all_language_probs = None
        if language is None:
            if not self.model.is_multilingual:
                language, language_probability = "en", 1
                all_language_probs = [("en", 1.0)]
            else:
                if feature_chunks:
                    (
                        language,
                        language_probability,
                        all_language_probs,
                    ) = self.detect_language(
                        features=feature_chunks[0],
                        language_detection_segments=language_detection_segments,
                        language_detection_threshold=language_detection_threshold,
                    )
                else:
                    language, language_probability = "en", 1
                    all_language_probs = [("en", 1.0)]

                self.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.is_multilingual and language != "en":
                self.logger.warning(
                    "English-only model but language='%s'; using 'en'.", language
                )
                language = "en"
            language_probability, all_language_probs = 1, None

        # --- tokenizer & options --------------------------------------------
        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )

        # --- transcribe each chunk ------------------------------------------
        segments = self.generate_segments_with_chunks(
            feature_chunks,
            speech_chunks,
            tokenizer,
            options,
            cross_chunk_prompt=cross_chunk_prompt,
            prompt_window_tokens=prompt_window_tokens,
            reset_prompt_on_gap_seconds=reset_prompt_on_gap_seconds,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        segments = self.restore_speech_timestamps_with_chunks(
            segments, speech_chunks, sampling_rate, log_progress=log_progress
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        return segments, info

    # -------------------------------------------------------------------
    def generate_segments_with_chunks(
        self,
        feature_chunks: List[np.ndarray],
        speech_chunks: List[Dict[str, int]],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        cross_chunk_prompt: bool = False,
        prompt_window_tokens: int = 64,
        reset_prompt_on_gap_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[Iterable[Segment]]:
        """Call ``self.generate_segments`` once per feature chunk with optional carryover."""
        segment_chunks: List[Iterable[Segment]] = []
        carryover_tokens: List[int] = []
        max_prompt_tokens = max(0, self.max_length // 2 - 1)
        requested_window = max(0, prompt_window_tokens)
        token_window = min(requested_window, max_prompt_tokens) if requested_window else 0

        for idx, feat in enumerate(feature_chunks):
            if cancel_event is not None and cancel_event.is_set():
                raise TaskCancelledError("任务已取消")

            if (
                cross_chunk_prompt
                and reset_prompt_on_gap_seconds is not None
                and idx > 0
                and idx < len(speech_chunks)
            ):
                gap_seconds = (
                    speech_chunks[idx]["start"] - speech_chunks[idx - 1]["end"]
                ) / self.feature_extractor.sampling_rate
                if gap_seconds > reset_prompt_on_gap_seconds:
                    carryover_tokens = []

            chunk_initial_prompt = options.initial_prompt if idx == 0 else None
            if cross_chunk_prompt and idx > 0 and carryover_tokens:
                chunk_initial_prompt = carryover_tokens.copy()

            chunk_options = dataclasses.replace(options, initial_prompt=chunk_initial_prompt)

            if progress_callback is not None:
                progress_callback(idx + 1, len(feature_chunks))

            chunk_segments = list(
                self.generate_segments(feat, tokenizer, chunk_options, log_progress=False)
            )
            segment_chunks.append(chunk_segments)

            if not cross_chunk_prompt:
                continue

            chunk_tokens = [
                token for seg in chunk_segments if seg.text.strip() for token in seg.tokens
            ]
            if not chunk_tokens:
                carryover_tokens = []
                continue

            carryover_tokens.extend(chunk_tokens)
            if token_window:
                carryover_tokens = carryover_tokens[-token_window:]
            elif max_prompt_tokens:
                carryover_tokens = carryover_tokens[-max_prompt_tokens:]

        return segment_chunks

    # -------------------------------------------------------------------
    def restore_speech_timestamps_with_chunks(
        self,
        segment_chunks: List[Iterable[Segment]],
        speech_chunks: List[Dict[str, int]],
        sampling_rate: int,
        log_progress: bool = False,
    ) -> Iterable[Segment]:
        """Map chunk-relative timestamps back to absolute audio positions."""
        total = len(speech_chunks)
        start_time = time.time()
        for idx, (segs, chunk) in enumerate(zip(segment_chunks, speech_chunks)):
            if log_progress:
                elapsed = time.time() - start_time
                elapsed_str = (
                    f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
                )
                if idx > 0:
                    eta = elapsed / idx * (total - idx)
                    eta_str = (
                        f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
                    )
                else:
                    eta_str = "--:--:--"

                chunk_start_sec = chunk["start"] / sampling_rate
                chunk_end_sec = chunk["end"] / sampling_rate
                chunk_range = (
                    f"[{format_timestamp(chunk_start_sec)} -> {format_timestamp(chunk_end_sec)}]"
                )

                print(
                    f"\rProcessing subtitle chunk {idx + 1}/{total} "
                    f"{chunk_range} | "
                    f"Elapsed: {elapsed_str} | "
                    f"ETA: {eta_str}",
                    end="",
                    flush=True,
                )

            chunk_start = chunk["start"] / sampling_rate
            chunk_end = chunk["end"] / sampling_rate

            for seg in segs:
                seg.start = max(chunk_start + seg.start, chunk_start)
                seg.end = min(chunk_start + seg.end, chunk_end)

                if seg.words:
                    for w in seg.words:
                        w.start = max(chunk_start, min(chunk_end, chunk_start + w.start))
                        w.end = max(chunk_start, min(chunk_end, chunk_start + w.end))

                yield seg

        if log_progress:
            print()