#!/usr/bin/env python3
"""
FireRedVAD ONNX 非流式语音活动检测 (VAD)

加载 fireredvad_vad.onnx 进行音频活动检测。
支持通过 ffmpeg 自动转码任意音频格式 (WAV/MP3/M4A/FLAC/...)。

作为模块导入:
    from main import vad_detect
    result = vad_detect("audio.wav")

直接运行:
    python main.py          # 使用下方 __main__ 中的配置
"""

import json
import logging
import sys
import os
import subprocess
from enum import Enum

import numpy as np
import onnxruntime as ort

from app.srt_utils import write_subtitle_file

# # 项目根目录 = 当前文件所在目录的父目录
# PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJ_DIR not in sys.path:
#     sys.path.insert(0, str(PROJ_DIR))


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)



# ---- 常量 ----
SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10
FRAME_SHIFT_SAMPLES = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)  # 160
FRAME_PER_SECOND = int(1000 / FRAME_SHIFT_MS)  # 100
FRAME_SHIFT_S = 0.010
FRAME_LENGTH_S = 0.025
FRAME_LENGTH_SAMPLES = int(SAMPLE_RATE * FRAME_LENGTH_S)
N_MELS = 80
N_FFT = 512


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * np.expm1(mel / 1127.0)


def povey_window(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    n = np.arange(length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))) ** 0.85


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_min = hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    max_bin = filterbank.shape[1] - 1
    for i in range(n_mels):
        left = int(np.clip(bin_points[i], 0, max_bin))
        center = int(np.clip(bin_points[i + 1], 0, max_bin))
        right = int(np.clip(bin_points[i + 2], 0, max_bin))

        if center <= left:
            center = min(left + 1, max_bin)
        if right <= center:
            right = min(center + 1, max_bin)
        if center <= left or right <= center:
            continue

        up = np.arange(left, center, dtype=np.int32)
        down = np.arange(center, right, dtype=np.int32)
        filterbank[i, up] = (up - left) / (center - left)
        filterbank[i, down] = (right - down) / (right - center)

    return filterbank


# ====================================================================
# 1. 音频加载
# ====================================================================


def load_audio_via_ffmpeg(input_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """调用 ffmpeg 将任意音频转为 16kHz 单声道 PCM int16，返回 numpy 数组。"""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sr),
        "-loglevel", "quiet", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {res.stderr.decode(errors='replace')}")

    # 标准 WAV 头 44 字节，跳过取 PCM 数据
    wav_data = res.stdout
    pcm_data = wav_data[44:]
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples


# ====================================================================
# 2. FBank 特征提取
# ====================================================================


class FbankExtractor:
    """纯 NumPy 80 维 FBank 特征提取。"""

    def __init__(self):
        self.frame_length = FRAME_LENGTH_SAMPLES
        self.frame_shift = FRAME_SHIFT_SAMPLES
        self.n_fft = N_FFT
        self.window = povey_window(self.frame_length)
        self.mel_filterbank = create_mel_filterbank(
            sample_rate=SAMPLE_RATE,
            n_fft=self.n_fft,
            n_mels=N_MELS,
        )
        self.remainder = np.array([], dtype=np.int16)

    def extract(self, pcm_int16: np.ndarray) -> np.ndarray | None:
        """提取 FBank 特征，返回 (T, 80) 或 None。"""
        samples = np.concatenate([self.remainder, pcm_int16])

        if len(samples) < self.frame_length:
            self.remainder = samples
            return None

        num_frames = 1 + (len(samples) - self.frame_length) // self.frame_shift
        if num_frames <= 0:
            self.remainder = samples
            return None

        total_length = (num_frames - 1) * self.frame_shift + self.frame_length
        used = samples[:total_length]
        consumed = num_frames * self.frame_shift
        self.remainder = samples[consumed:]

        frames = np.lib.stride_tricks.sliding_window_view(used, self.frame_length)[
            :: self.frame_shift
        ]
        frames = frames.astype(np.float32, copy=True)
        frames *= self.window

        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power = np.abs(spectrum).astype(np.float32) ** 2
        mel = power @ self.mel_filterbank.T
        mel = np.maximum(mel, 1e-10)
        return np.log(mel).astype(np.float32)

    def reset(self):
        self.remainder = np.array([], dtype=np.int16)


# ====================================================================
# 3. CMVN 加载
# ====================================================================


def load_cmvn(cmvn_path: str) -> dict:
    import struct
    import numpy as np

    def read_kaldi_int(data, offset):
        size = data[offset]
        offset += 1
        if size != 4:
            raise ValueError(f"Unsupported int size: {size}")
        val = struct.unpack_from("<i", data, offset)[0]
        offset += size
        return val, offset

    with open(cmvn_path, "rb") as f:
        data = f.read()

    # 找到 \0B
    start = data.find(b'\x00B')
    if start == -1:
        raise ValueError("Invalid CMVN: no \\0B header")

    offset = start + 2

    # dtype
    dtype_char = data[offset:offset+1]
    offset += 1

    if data[offset:offset+1] != b'M':
        raise ValueError("Invalid CMVN format")
    offset += 1

    if data[offset:offset+1] != b' ':
        raise ValueError("Invalid CMVN format")
    offset += 1

    # ✅ 正确读取 rows / cols
    rows, offset = read_kaldi_int(data, offset)
    cols, offset = read_kaldi_int(data, offset)

    # dtype
    if dtype_char == b'D':
        np_dtype = np.float64
    elif dtype_char == b'F':
        np_dtype = np.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_char}")

    # 读取矩阵
    count = rows * cols
    stats = np.frombuffer(
        data,
        dtype=np_dtype,
        count=count,
        offset=offset
    ).reshape(rows, cols)

    # 计算 CMVN
    dim = cols - 1
    total = stats[0, dim]

    if total <= 0:
        raise ValueError("Invalid CMVN count")

    means = stats[0, :dim] / total
    variances = (stats[1, :dim] / total) - means ** 2
    variances = np.maximum(variances, 1e-20)
    inv_std = 1.0 / np.sqrt(variances)

    print(f"✅ CMVN loaded: shape={stats.shape}, dtype={np_dtype}")

    return {
        "means": means.astype(np.float32),
        "inv_std": inv_std.astype(np.float32),
    }


# ====================================================================
# 4. VAD 后处理
# ====================================================================


class VadState(Enum):
    SILENCE = 0
    POSSIBLE_SPEECH = 1
    SPEECH = 2
    POSSIBLE_SILENCE = 3


class VadPostprocessor:
    """VAD 后处理：平滑 → 阈值 → 状态机 → 短静音合并 → 扩展 → 超长分割。"""

    def __init__(
        self,
        smooth_window_size: int = 5,
        prob_threshold: float = 0.4,
        min_speech_ms: int = 200,
        max_speech_ms: int = 20000,
        min_silence_ms: int = 200,
        merge_silence_ms: int = 0,
        extend_speech_ms: int = 0,
    ):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = prob_threshold
        # 内部以 frame 为单位处理，外部参数使用 ms
        self.min_speech_frame = min_speech_ms // FRAME_SHIFT_MS
        self.max_speech_frame = max_speech_ms // FRAME_SHIFT_MS
        self.min_silence_frame = min_silence_ms // FRAME_SHIFT_MS
        self.merge_silence_frame = merge_silence_ms // FRAME_SHIFT_MS
        self.extend_speech_frame = extend_speech_ms // FRAME_SHIFT_MS

    # ---- 主入口 ----

    def process(self, raw_probs: list[float]) -> list[int]:
        """输入帧级概率，输出 0/1 决策序列。"""
        if not raw_probs:
            return []

        smoothed = self._smooth_prob(raw_probs)
        binary = self._apply_threshold(smoothed)
        decisions = self._smooth_preds_with_state_machine(binary)
        decisions = self._fix_smooth_window_start(decisions)
        decisions = self._merge_short_silence_segments(decisions)
        decisions = self._extend_speech_segments(decisions)
        decisions = self._split_long_speech_segments(decisions, raw_probs)
        return decisions

    def decision_to_segment(
        self, decisions: list[int], wav_dur: float | None = None
    ) -> list[tuple[float, float]]:
        """将 0/1 决策序列转换为 (start_s, end_s) 列表。"""
        segments: list[tuple[float, float]] = []
        speech_start: int | None = None
        for t, d in enumerate(decisions):
            if d == 1 and speech_start is None:
                speech_start = t
            elif d == 0 and speech_start is not None:
                dur_frames = t - speech_start
                if dur_frames < self.min_speech_frame:
                    logger.warning(
                        "Unexpected short speech segment (%d frames), "
                        "check VadPostprocessor", dur_frames,
                    )
                segments.append((speech_start * FRAME_SHIFT_S, t * FRAME_SHIFT_S))
                speech_start = None
        if speech_start is not None:
            t = len(decisions) - 1
            dur_frames = t - speech_start
            if dur_frames < self.min_speech_frame:
                logger.warning(
                    "Unexpected short speech segment (%d frames), "
                    "check VadPostprocessor", dur_frames,
                )
            end_time = len(decisions) * FRAME_SHIFT_S + FRAME_LENGTH_S
            if wav_dur is not None:
                end_time = min(end_time, wav_dur)
            segments.append((speech_start * FRAME_SHIFT_S, end_time))
        segments = [(round(s, 3), round(e, 3)) for s, e in segments]
        return segments

    # ---- 内部方法 ----

    def _smooth_prob(self, probs: list[float]) -> np.ndarray:
        """滑动窗口平滑概率。"""
        if self.smooth_window_size <= 1:
            return np.asarray(probs, dtype=np.float64)
        probs_np = np.array(probs, dtype=np.float64)
        kernel = np.ones(self.smooth_window_size) / self.smooth_window_size
        smoothed = np.convolve(probs_np, kernel, mode="full")[: len(probs)]
        # 前几帧用累积平均
        for i in range(min(self.smooth_window_size - 1, len(probs))):
            smoothed[i] = np.mean(probs_np[: i + 1])
        return smoothed

    def _apply_threshold(self, probs: np.ndarray) -> list[int]:
        return (probs >= self.prob_threshold).astype(int).tolist()

    def _smooth_preds_with_state_machine(self, binary_preds: list[int]) -> list[int]:
        """状态机：约束最小语音帧和最小静音帧。"""
        if self.min_speech_frame <= 0 and self.min_silence_frame <= 0:
            return binary_preds

        decisions = [0] * len(binary_preds)
        state = VadState.SILENCE
        speech_start = -1
        silence_start = -1

        for t, is_speech in enumerate(binary_preds):
            # ---- 状态转移 ----
            if state == VadState.SILENCE:
                if is_speech:
                    state = VadState.POSSIBLE_SPEECH
                    speech_start = t

            elif state == VadState.POSSIBLE_SPEECH:
                if is_speech:
                    assert speech_start != -1
                    if t - speech_start >= self.min_speech_frame:
                        state = VadState.SPEECH
                        decisions[speech_start:t] = [1] * (t - speech_start)
                else:
                    state = VadState.SILENCE
                    speech_start = -1

            elif state == VadState.SPEECH:
                if not is_speech:
                    state = VadState.POSSIBLE_SILENCE
                    silence_start = t

            elif state == VadState.POSSIBLE_SILENCE:
                if not is_speech:
                    assert silence_start != -1
                    if t - silence_start >= self.min_silence_frame:
                        state = VadState.SILENCE
                        speech_start = -1
                else:
                    state = VadState.SPEECH
                    silence_start = -1

            # ---- 当前帧决策 ----
            if state in (VadState.SPEECH, VadState.POSSIBLE_SILENCE):
                decision = 1
            else:
                decision = 0

            decisions[t] = decision

        return decisions

    def _fix_smooth_window_start(self, decisions: list[int]) -> list[int]:
        """将语音段起始前 smooth_window_size 帧也标记为语音（补偿平滑延迟）。"""
        new_decisions = decisions.copy()
        for t in range(1, len(decisions)):
            if decisions[t - 1] == 0 and decisions[t] == 1:
                start = max(0, t - self.smooth_window_size)
                new_decisions[start:t] = [1] * (t - start)
        return new_decisions

    def _merge_short_silence_segments(self, decisions: list[int]) -> list[int]:
        """合并小于 merge_silence_frame 的静音间隔。"""
        if self.merge_silence_frame <= 0:
            return decisions
        new_decisions = decisions.copy()
        silence_start: int | None = None
        for t in range(1, len(decisions)):
            if decisions[t - 1] == 1 and decisions[t] == 0 and silence_start is None:
                silence_start = t
            elif decisions[t - 1] == 0 and decisions[t] == 1 and silence_start is not None:
                silence_frames = t - silence_start
                if silence_frames < self.merge_silence_frame:
                    new_decisions[silence_start:t] = [1] * silence_frames
                silence_start = None
        return new_decisions

    def _extend_speech_segments(self, decisions: list[int]) -> list[int]:
        """用卷积扩展语音段边界。"""
        if self.extend_speech_frame <= 0:
            return decisions
        decisions_np = np.array(decisions, dtype=np.float64)
        kernel = np.ones(2 * self.extend_speech_frame + 1)
        extended = np.convolve(decisions_np, kernel, mode="same")
        return (extended > 0).astype(int).tolist()

    def _split_long_speech_segments(self, decisions: list[int], probs: list[float]) -> list[int]:
        """将超过 max_speech_frame 的语音段在概率最低点分割。"""
        new_decisions = decisions.copy()
        segments = self.decision_to_segment(decisions)
        for start_s, end_s in segments:
            start_frame = int(start_s / FRAME_SHIFT_S)
            end_frame = int(end_s / FRAME_SHIFT_S)
            dur_frames = end_frame - start_frame
            if dur_frames > self.max_speech_frame:
                segment_probs = probs[start_frame:end_frame]
                split_points = self._find_split_points(segment_probs)
                for sp in split_points:
                    split_frame = start_frame + sp
                    new_decisions[split_frame] = 0
        return new_decisions

    def _find_split_points(self, probs: list[float]) -> list[int]:
        split_points: list[int] = []
        length = len(probs)
        start = 0
        while start < length:
            if (length - start) <= self.max_speech_frame:
                break
            window_start = start + self.max_speech_frame // 2
            window_end = start + self.max_speech_frame
            window_probs = probs[window_start:window_end]
            min_index = window_start + int(np.argmin(window_probs))
            split_points.append(min_index)
            start = min_index + 1
        return split_points


class CleanVadPostprocessor:
    """
    干净版 VAD 后处理：
    - 双阈值防抖
    - 单一 merge 控制切句
    - 最小语音段通过合并保证长度
    - 最长语音限制 (max_speech_ms)
    - 无状态机
    """

    def __init__(
        self,
        smooth_window_size: int = 5,
        start_threshold: float = 0.6,
        end_threshold: float = 0.3,
        min_speech_ms: int = 150,
        max_speech_ms: int | None = 20000,
        merge_silence_ms: int = 300,
        extend_speech_ms: int = 50,
    ):
        self.smooth_window_size = max(1, smooth_window_size)

        # 双阈值（关键）
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.min_speech_frame = min_speech_ms // FRAME_SHIFT_MS
        if max_speech_ms is not None and max_speech_ms > 0:
            self.max_speech_frame: int | None = max(1, max_speech_ms // FRAME_SHIFT_MS)
        else:
            self.max_speech_frame = None
        self.merge_silence_frame = merge_silence_ms // FRAME_SHIFT_MS
        self.extend_speech_frame = extend_speech_ms // FRAME_SHIFT_MS

    # =========================================================
    # 主入口
    # =========================================================

    def process(self, probs: list[float]) -> list[int]:
        if not probs:
            return []

        smoothed_probs = self._smooth(probs)
        binary = self._hysteresis_threshold(smoothed_probs)
        binary = self._merge_silence(binary)
        binary = self._enforce_min_speech(binary)
        binary = self._extend(binary)
        binary = self._split_long_speech_segments(binary, smoothed_probs)

        return binary

    # =========================================================
    # 1. 平滑
    # =========================================================

    def _smooth(self, probs):
        if self.smooth_window_size <= 1:
            return np.array(probs, dtype=np.float32)

        kernel = np.ones(self.smooth_window_size) / self.smooth_window_size
        smoothed = np.convolve(probs, kernel, mode="same")
        return smoothed

    # =========================================================
    # 2. 双阈值（关键稳定器）
    # =========================================================

    def _hysteresis_threshold(self, probs):
        result = []
        state = 0  # 0=silence, 1=speech

        for p in probs:
            if state == 0:
                if p >= self.start_threshold:
                    state = 1
            else:
                if p < self.end_threshold:
                    state = 0

            result.append(state)

        return result

    # =========================================================
    # 3. merge 静音（唯一切句逻辑）
    # =========================================================

    def _merge_silence(self, binary):
        if self.merge_silence_frame <= 0:
            return binary

        result = binary.copy()
        silence_start = None

        for i in range(1, len(binary)):
            if binary[i-1] == 1 and binary[i] == 0:
                silence_start = i

            elif binary[i-1] == 0 and binary[i] == 1 and silence_start is not None:
                silence_len = i - silence_start

                if silence_len < self.merge_silence_frame:
                    result[silence_start:i] = [1] * silence_len

                silence_start = None

        return result

    # =========================================================
    # 4. 去掉短语音（只做一次）
    # =========================================================

    def _enforce_min_speech(self, binary: list[int]) -> list[int]:
        if self.min_speech_frame <= 0:
            return binary

        decisions = binary.copy()
        total_frames = len(decisions)
        if total_frames == 0:
            return decisions

        while True:
            segments = self._find_speech_segments(decisions)
            changed = False

            for idx, (seg_start, seg_end) in enumerate(segments):
                seg_len = seg_end - seg_start
                if seg_len >= self.min_speech_frame:
                    continue

                # Gather consecutive short segments and the silences between them.
                group_start_idx = idx
                group_end_idx = idx
                accumulated = seg_len

                while (
                    group_end_idx + 1 < len(segments)
                    and (segments[group_end_idx + 1][1] - segments[group_end_idx + 1][0]) < self.min_speech_frame
                ):
                    gap_start = segments[group_end_idx][1]
                    gap_end = segments[group_end_idx + 1][0]
                    accumulated += (gap_end - gap_start)
                    accumulated += segments[group_end_idx + 1][1] - segments[group_end_idx + 1][0]
                    group_end_idx += 1

                merge_start = segments[group_start_idx][0]
                merge_end = segments[group_end_idx][1]

                if accumulated >= self.min_speech_frame:
                    decisions[merge_start:merge_end] = [1] * (merge_end - merge_start)
                    changed = True
                    break

                # Try merging with adjacent longer segments.
                candidates: list[tuple[int, int, int]] = []
                if group_start_idx > 0:
                    prev_start, prev_end = segments[group_start_idx - 1]
                    total_len = merge_end - prev_start
                    candidates.append((prev_start, merge_end, total_len))
                if group_end_idx + 1 < len(segments):
                    next_start, next_end = segments[group_end_idx + 1]
                    total_len = next_end - merge_start
                    candidates.append((merge_start, next_end, total_len))

                best: tuple[int, int, int] | None = None
                for cand in candidates:
                    if cand[2] >= self.min_speech_frame:
                        if best is None or cand[2] < best[2]:
                            best = cand
                if best is None and candidates:
                    best = max(candidates, key=lambda c: c[2])

                if best is not None:
                    start_idx, end_idx, _ = best
                    decisions[start_idx:end_idx] = [1] * (end_idx - start_idx)
                    changed = True
                    break

                # As a fallback, extend the current merged block until reaching min length or bounds.
                deficit = self.min_speech_frame - (merge_end - merge_start)
                pad_left = min(merge_start, deficit // 2)
                pad_right = min(total_frames - merge_end, deficit - pad_left)
                merge_start -= pad_left
                merge_end += pad_right
                decisions[merge_start:merge_end] = [1] * (merge_end - merge_start)
                changed = True
                break

            if not changed:
                break

        return decisions

    @staticmethod
    def _find_speech_segments(decisions: list[int]) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start: int | None = None

        for idx, value in enumerate(decisions):
            if value == 1:
                if start is None:
                    start = idx
            elif start is not None:
                segments.append((start, idx))
                start = None

        if start is not None:
            segments.append((start, len(decisions)))

        return segments

    # =========================================================
    # 5. 扩展边界（可选）
    # =========================================================

    def _extend(self, binary):
        if self.extend_speech_frame <= 0:
            return binary

        binary_np = np.array(binary)
        kernel = np.ones(2 * self.extend_speech_frame + 1)
        extended = np.convolve(binary_np, kernel, mode="same")

        return (extended > 0).astype(int).tolist()

    # =========================================================
    # 6. 限制最长语音段
    # =========================================================

    def _split_long_speech_segments(self, decisions, probs):
        if self.max_speech_frame is None:
            return decisions

        new_decisions = decisions.copy()
        segments = self.decision_to_segment(decisions)
        if not segments:
            return new_decisions

        probs_array = np.asarray(probs, dtype=np.float32)

        for start_s, end_s in segments:
            start_frame = int(start_s / FRAME_SHIFT_S)
            end_frame = int(end_s / FRAME_SHIFT_S)
            dur_frames = end_frame - start_frame
            if dur_frames > self.max_speech_frame:
                segment_probs = probs_array[start_frame:end_frame]
                split_points = self._find_split_points(segment_probs)
                for sp in split_points:
                    split_frame = start_frame + sp
                    if 0 <= split_frame < len(new_decisions):
                        new_decisions[split_frame] = 0

        return new_decisions

    def _find_split_points(self, probs):
        split_points: list[int] = []
        length = len(probs)
        start = 0
        while start < length:
            if (length - start) <= self.max_speech_frame:
                break
            window_start = start + self.max_speech_frame // 2
            window_end = start + self.max_speech_frame
            window_probs = probs[window_start:window_end]
            if len(window_probs) == 0:
                break
            min_index = int(window_start + np.argmin(window_probs))
            split_points.append(min_index)
            start = min_index + 1
        return split_points

    # =========================================================
    # 转时间段
    # =========================================================

    def decision_to_segment(self, decisions, wav_dur=None):
        segments = []
        start = None

        for i, d in enumerate(decisions):
            if d == 1 and start is None:
                start = i

            elif d == 0 and start is not None:
                segments.append((
                    start * FRAME_SHIFT_S,
                    i * FRAME_SHIFT_S
                ))
                start = None

        if start is not None:
            end_time = len(decisions) * FRAME_SHIFT_S + FRAME_LENGTH_S
            if wav_dur is not None:
                end_time = min(end_time, wav_dur)
            segments.append((
                start * FRAME_SHIFT_S,
                end_time
            ))

        return [(round(s, 3), round(e, 3)) for s, e in segments]


# ====================================================================
# 5. ONNX VAD 主类
# ====================================================================


class OnnxVad:
    """基于 ONNX 的非流式 VAD。"""

    def __init__(
        self,
        model_path: str,
        cmvn_path: str,
        **vad_kwargs
    ):
        logger.info("Available providers: %s", ort.get_available_providers())
        logger.info("Loading ONNX model: %s", model_path)
        try:
            self.sess = ort.InferenceSession(
                model_path,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("Using DML execution provider")
        except Exception as e:
            logger.warning("DML unavailable, fallback to CPU: %s", e)
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.fbank = FbankExtractor()
        logger.info("Loading CMVN: %s", cmvn_path)
        self.cmvn = load_cmvn(cmvn_path)

        self.postprocessor = CleanVadPostprocessor(**vad_kwargs)

        # ONNX 模型输入名
        self.input_name = self.sess.get_inputs()[0].name
        logger.info("Model input: %s", self.input_name)

    def detect(self, pcm_int16: np.ndarray) -> dict:
        """
        对整段 PCM int16 音频进行 VAD 检测。

        返回:
            {
                "dur": float,          # 音频时长 (秒)
                "timestamps": [...],   # [(start_s, end_s), ...]
                "probs": [...]         # 帧级概率 (可选)
            }
        """
        dur = len(pcm_int16) / SAMPLE_RATE
        logger.info("Audio duration: %.3f s", dur)

        # 1. 提取 FBank
        fbank = self.fbank.extract(pcm_int16)
        if fbank is None or len(fbank) == 0:
            logger.warning("No frames extracted from audio")
            return {"dur": round(dur, 3), "timestamps": [], "probs": []}

        # 2. CMVN
        fbank = (fbank - self.cmvn["means"]) * self.cmvn["inv_std"]  # (T, 80)

        # 3. ONNX 推理
        feat = fbank.astype(np.float32)[np.newaxis, :, :]  # (1, T, 80)
        (probs,) = self.sess.run(None, {self.input_name: feat})
        probs = probs.squeeze()  # (T,)
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        probs_list = probs.tolist()
        logger.info("Inference done, %d frames", len(probs_list))

        # 4. 后处理
        decisions = self.postprocessor.process(probs_list)
        timestamps = self.postprocessor.decision_to_segment(decisions, dur)
        logger.info("Detected %d speech segments", len(timestamps))
        for s, e in timestamps:
            logger.info("  %.3f - %.3f (%.3f s)", s, e, e - s)

        return {
            "dur": round(dur, 3),
            "timestamps": timestamps,
            "probs": [round(p, 6) for p in probs_list],
        }


# ====================================================================
# 7. 公开 API — 外部程序可调用
# ====================================================================


def vad_detect(
    wav_path: str,
    model_path: str = "fireredvad_vad.onnx",
    cmvn_path: str = "cmvn.ark",
    output: str | None = None,
    output_srt: str | None = None,
    smooth_window_size: int = 3,
    start_threshold: float = 0.6,
    end_threshold: float = 0.3,
    min_speech_ms: int = 150,
    max_speech_ms: int | None = 20000,
    merge_silence_ms: int = 350,
    extend_speech_ms: int = 50,
    verbose: bool = False,
) -> dict:
    """
    对音频文件执行非流式 VAD 检测。

    参数:
        wav_path: 音频文件路径 (支持 WAV/MP3/M4A/FLAC/...)
        model_path: ONNX 模型路径
        cmvn_path: CMVN 统计文件路径
        output: JSON 结果保存路径 (None 则不保存)
        output_srt: SRT 字幕文件路径 (None 则自动生成)
        smooth_window_size: 平滑窗口大小
        start_threshold: 语音起始阈值
        end_threshold: 语音结束阈值
        min_speech_ms: 最小语音段长度 (ms)，短片段会自动合并以满足该长度
        max_speech_ms: 最长语音段长度 (ms)，超过此长度会自动切分；None 表示不限制
        merge_silence_ms: 合并静音长度 (ms)
        extend_speech_ms: 边界扩展长度 (ms)
        verbose: 是否输出详细日志

    返回:
        {
            "wav_path": str,
            "dur": float,           # 音频时长 (秒)
            "timestamps": [...],    # [(start_s, end_s), ...]
            "probs": [...]          # 帧级概率
        }
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查文件是否存在
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"音频文件不存在: {wav_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(cmvn_path):
        raise FileNotFoundError(f"CMVN 文件不存在: {cmvn_path}")

    # 创建 VAD 引擎
    vad_config = {
        "smooth_window_size": smooth_window_size,
        "start_threshold": start_threshold,
        "end_threshold": end_threshold,
        "min_speech_ms": min_speech_ms,  # 最短语音长度 (ms)，内部会自动转为帧数
        "max_speech_ms": max_speech_ms,
        "merge_silence_ms": merge_silence_ms,
        "extend_speech_ms": extend_speech_ms,
    }
    vad = OnnxVad(model_path=model_path, cmvn_path=cmvn_path, **vad_config)

    logger.info("=" * 50)
    logger.info("Processing: %s", wav_path)

    # 加载音频
    pcm = load_audio_via_ffmpeg(wav_path)

    # 执行 VAD 检测
    result = vad.detect(pcm)
    result["wav_path"] = wav_path

    # 保存 JSON 结果
    if output:
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("JSON 结果已保存到 %s", output)

    # 保存 SRT 字幕
    if result["timestamps"]:
        if output_srt is None:
            if output:
                base, _ = os.path.splitext(output)
                output_srt = base + ".srt"
            else:
                base, _ = os.path.splitext(wav_path)
                output_srt = base + ".srt"
        write_subtitle_file(
            path=output_srt, 
            timestamps=result["timestamps"], 
            wav_dur=result.get("dur")
            )
        # 验证 SRT 文件确实写入了内容
        if os.path.isfile(output_srt):
            file_size = os.path.getsize(output_srt)
            if file_size < 10:
                logger.warning(
                    "SRT 文件 %s 大小仅 %d 字节，可能写入异常！",
                    output_srt, file_size,
                )
        logger.info("SRT 字幕已保存到 %s", output_srt)
    else:
        logger.info("无语音段，跳过 SRT 生成")

    return result


def vad_detect_simple(
    pcm: np.ndarray,
    model_path: str = "fireredvad_vad.onnx",
    cmvn_path: str = "cmvn.ark",
    smooth_window_size: int = 3,
    start_threshold: float = 0.6,
    end_threshold: float = 0.3,
    min_speech_ms: int = 150,
    max_speech_ms: int | None = 20000,
    merge_silence_ms: int = 350,
    extend_speech_ms: int = 50,
) -> list[tuple[float, float]]:
    """
    对 PCM int16 音频数据执行非流式 VAD 检测，仅返回时间戳列表。

    参数:
        pcm: PCM int16 音频数据 (numpy 数组)
        model_path: ONNX 模型路径
        cmvn_path: CMVN 统计文件路径
        smooth_window_size: 平滑窗口大小
        start_threshold: 语音起始阈值
        end_threshold: 语音结束阈值
        min_speech_ms: 最小语音段长度 (ms)，短片段会自动合并以满足该长度
        max_speech_ms: 最长语音段长度 (ms)，超过此长度会自动切分；None 表示不限制
        merge_silence_ms: 合并静音长度 (ms)
        extend_speech_ms: 边界扩展长度 (ms)

    返回:
        [(start_s, end_s), ...]  # 语音段时间戳列表 (秒)
    """
    # 检查模型和 CMVN 文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(cmvn_path):
        raise FileNotFoundError(f"CMVN 文件不存在: {cmvn_path}")

    # 创建 VAD 引擎
    vad_config = {
        "smooth_window_size": smooth_window_size,
        "start_threshold": start_threshold,
        "end_threshold": end_threshold,
        "min_speech_ms": min_speech_ms,
        "max_speech_ms": max_speech_ms,
        "merge_silence_ms": merge_silence_ms,
        "extend_speech_ms": extend_speech_ms,
    }
    vad = OnnxVad(model_path=model_path, cmvn_path=cmvn_path, **vad_config)

    # 执行 VAD 检测
    result = vad.detect(pcm)

    return result["timestamps"]


if __name__ == "__main__":
    # ================================================================
    # 使用示例 — 在此修改参数运行
    # ================================================================
    result = vad_detect(
        wav_path=r"path\to\input.wav",
        output_srt=r"result.srt",
        smooth_window_size=3,
        start_threshold=0.6,
        end_threshold=0.3,
        min_speech_ms=2000,
        max_speech_ms=30000,
        merge_silence_ms=350,
        extend_speech_ms=50,
        verbose=False,
    )

    # 打印结果摘要
    print(f"\n音频文件: {result['wav_path']}")
    print(f"时长: {result['dur']:.3f}s")
    if result["timestamps"]:
        print(f"语音段 ({len(result['timestamps'])} 个)")
        for s, e in result["timestamps"]:
            print(f"    {s:.3f} - {e:.3f}  ({e - s:.3f}s)")
    else:
        print("未检测到语音")
