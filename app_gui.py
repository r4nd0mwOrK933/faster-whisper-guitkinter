"""
Faster-Whisper GUI application with Chinese and English interfaces.
Tkinter-based VAD → ASR pipeline with drag-and-drop file support.
"""

from __future__ import annotations

import configparser
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from shutil import copyfile

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

from app.subtitle_transcribe import TaskCancelledError, WhisperSubtitle, load_cudnn
from app.srt_utils import write_srt_file
from app.localization import LANGUAGE_CODES, LANGUAGE_LABELS, TRANSLATIONS


# 本项目使用 CUDA 12 的 faster-whisper 运行库。只检查目录中存在任意 DLL
# 会让 ctypes 在加载模型时才失败，因此在启动 ASR 前明确检查这些关键文件。
REQUIRED_CUDA_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn_ops_infer64_8.dll",
    "zlibwapi.dll",
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.json",
)


# ============================================================================
# 任务参数快照
# ============================================================================

@dataclass(frozen=True)
class TaskSettings:
    """启动任务时从界面冻结的不可变参数快照。

    后台线程只读取此对象，绝不直接访问任何 Tk 控件；
    界面更新一律通过事件队列（root.after）回到主线程完成。
    """

    # ---- 路径 ----
    audio_path: str
    model_dir: str
    subtitle_path: Optional[str]  # 手动运行 ASR 时的字幕来源
    cudnn_dir: str

    # ---- 设备 ----
    device: str          # "cuda" | "cpu"
    compute_type: str

    # ---- VAD ----
    start_threshold: float
    end_threshold: float
    min_speech_ms: int
    max_speech_ms: int
    merge_silence_ms: int
    extend_speech_ms: int

    # ---- ASR ----
    language: Optional[str]  # None 表示自动检测
    initial_prompt: Optional[str]
    hotwords: Optional[str]
    beam_size: int
    best_of: int
    word_timestamps: bool
    log_progress: bool
    gui_language: str
    overwrite_if_exists: bool = False

    # ---- 派生路径 ----
    @property
    def vad_srt_path(self) -> str:
        """VAD 输出 SRT（音频同目录，{stem}_vad.srt）。"""
        audio_p = Path(self.audio_path)
        return str(audio_p.parent / f"{audio_p.stem}_vad.srt")

    def asr_srt_path(self, subtitle_path: Optional[str] = None) -> str:
        """ASR 输出 SRT（使用字幕文件名基础，保存到音频所在目录）。"""
        audio_p = Path(self.audio_path)
        subtitle_p = Path(subtitle_path or self.subtitle_path or audio_p.stem)
        return str(audio_p.parent / f"{subtitle_p.stem}_asr.srt")


# ============================================================================
# 配置管理器
# ============================================================================

class ConfigManager:
    """读写 gui_config.ini，为所有小节提供默认值。"""

    DEFAULTS = {
        "PATHS": {
            "model_dir": "",
            "audio_path": "",
            "subtitle_path": "",
            "cudnn_dir": "",
        },
        "GUI": {
            "language": "zh",
        },
        "WHISPER": {
            "log_progress": "true",
            "initial_prompt": "",
            "language": "ja",
            "hotwords": "",
            "beam_size": "5",
            "best_of": "3",
            "word_timestamps": "false",
            "allow_subtitle_mismatch": "false",
            "overwrite_if_exists": "false",
        },
        "VAD": {
            "start_threshold": "0.6",
            "end_threshold": "0.3",
            "min_speech_ms": "2000",
            "max_speech_ms": "30000",
            "merge_silence_ms": "350",
            "extend_speech_ms": "50",
        },
    }

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = configparser.ConfigParser()

    def load(self):
        """加载配置，缺失的小节/键用默认值补齐。"""
        if not self.config_path.exists():
            template_path = self.config_path.with_name("gui_config.example.ini")
            if template_path.exists():
                copyfile(template_path, self.config_path)
                
        if self.config_path.exists():
            self.config.read(self.config_path, encoding="utf-8")
        self._ensure_defaults()

    def save(self):
        """将配置写入文件。"""
        self._ensure_defaults()
        with open(self.config_path, "w", encoding="utf-8") as f:
            self.config.write(f)

    def _ensure_defaults(self):
        for section, items in self.DEFAULTS.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for key, value in items.items():
                if key not in self.config[section]:
                    self.config[section][key] = value

    def get(self, section: str, key: str, default: str = ""):
        """获取原始字符串值（去除引号）。"""
        try:
            value = self.config[section].get(key, fallback=default)
        except configparser.Error:
            return default
        if value is None:
            return default
        return value.strip().strip('"').strip("'")

    def get_float(self, section: str, key: str, default: float = 0.0):
        try:
            return self.config.getfloat(section, key, fallback=default)
        except (ValueError, configparser.Error):
            return default

    def get_int(self, section: str, key: str, default: int = 0):
        try:
            return self.config.getint(section, key, fallback=default)
        except (ValueError, configparser.Error):
            return default

    def get_bool(self, section: str, key: str, default: bool = False):
        try:
            return self.config.getboolean(section, key, fallback=default)
        except (ValueError, configparser.Error):
            return default

    def set(self, section: str, key: str, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config[section][key] = str(value)


# ============================================================================
# 拖放输入框控件
# ============================================================================

class DragDropEntry(ttk.Frame):
    """带“浏览”按钮的输入框，可选支持文件拖放。

    Args:
        mode: "file" 表示文件选择器，"folder" 表示目录选择器。
    """

    def __init__(self, parent, mode: str = "file", **kwargs):
        """
        Args:
            mode: "file" 表示文件选择器，"folder" 表示目录选择器。
        """
        super().__init__(parent)
        self.mode = mode
        self.var = tk.StringVar()
        self._trace_id = self.var.trace_add("write", self._strip_quotes)

        self.entry = ttk.Entry(self, textvariable=self.var, **kwargs)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ttk.Button(
            self, text="浏览", command=self._browse, width=8
        )
        self.browse_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._dialog_texts = {
            "select_folder": "选择文件夹",
            "select_file": "选择文件",
            "media_files": "媒体文件",
            "subtitle_files": "字幕文件",
            "all_files": "所有文件",
        }

        if HAS_DND:
            self.entry.drop_target_register(DND_FILES)
            self.entry.dnd_bind("<<Drop>>", self._on_drop)

    # ------------------------------------------------------------------
    def _on_drop(self, event):
        """从 Windows 拖放事件中提取第一个文件路径。"""
        data = event.data
        if not data:
            return
        data = data.strip()
        if data.startswith("{"):
            close_idx = data.find("}")
            path = data[1:close_idx] if close_idx > 0 else data[1:]
        else:
            path = data.split()[0] if data else ""
        path = path.strip().strip('"').strip("'")
        if path:
            self.var.set(path)

    # ------------------------------------------------------------------
    def _strip_quotes(self, *_):
        """自动去除输入框值两侧的双引号/单引号。"""
        value = self.var.get()
        if not value:
            return
        stripped = value.strip()
        if len(stripped) >= 2:
            if (stripped[0] == stripped[-1]) and stripped[0] in ('"', "'"):
                stripped = stripped[1:-1]
        if stripped != value:
            self.var.trace_remove("write", self._trace_id)
            self.var.set(stripped)
            self._trace_id = self.var.trace_add("write", self._strip_quotes)

    # ------------------------------------------------------------------
    def _browse(self):
        """根据 mode 打开文件或文件夹对话框。"""
        texts = self._dialog_texts
        if self.mode == "folder":
            result = filedialog.askdirectory(title=texts["select_folder"])
        else:
            result = filedialog.askopenfilename(
                title=texts["select_file"],
                filetypes=[
                    (texts["media_files"], "*.mp4 *.wav *.mp3 *.m4a *.flac *.mkv *.avi *.mov *.webm"),
                    (texts["subtitle_files"], "*.srt *.ass"),
                    (texts["all_files"], "*.*"),
                ],
            )
        if result:
            self.var.set(result)

    # ------------------------------------------------------------------
    def get(self):
        return self.var.get().strip()

    def set(self, value):
        self.var.set(str(value))

    def set_browse_text(self, text: str):
        self.browse_btn.configure(text=text)

    def set_dialog_texts(self, texts: dict[str, str]):
        self._dialog_texts = texts


# ============================================================================
# 主应用程序
# ============================================================================

class App:
    """Faster-Whisper VAD → ASR 流水线主图形界面。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._gui_language = "zh"
        self._translatable_widgets: dict[str, tk.Widget] = {}
        self.root.geometry("820x680")
        self.root.minsize(820, 680)
        
        self._set_app_icon()

        # ---- 配置 ----
        self.config_path = Path(__file__).resolve().parent / "gui_config.ini"
        self.config = ConfigManager(self.config_path)
        self.config.load()

        # ---- 状态 ----
        self.vad_srt_path: Optional[str] = None
        self.running = False
        self._cancel_event = threading.Event()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.progress_queue: queue.Queue[str] = queue.Queue()
        # 日志内容缓存（Python 侧管理，避免依赖 Tk 索引算术）
        self._log_lines: list[str] = []
        self._progress_text: Optional[str] = None

        # 模型缓存 — 在多次 ASR 运行之间将模型保留在内存中
        self._model = None          # WhisperSubtitle 实例或 None
        self._model_key = None      # tuple(model_dir, device, compute_type)

        # ---- 构建界面，然后从配置填充 ----
        self._build_ui()
        self.root.bind_all("<Button-1>", self._clear_entry_focus, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_config_to_ui()
        self._poll_log()

    def _translate(self, gui_language: str, key: str, **kwargs) -> str:
        gui_language = gui_language if gui_language in LANGUAGE_CODES else "zh"
        text = TRANSLATIONS[gui_language][key]
        return text.format(**kwargs) if kwargs else text

    def _tr(self, key: str, **kwargs) -> str:
        return self._translate(self._gui_language, key, **kwargs)

    def _register_text_widget(self, key: str, widget: tk.Widget):
        self._translatable_widgets[key] = widget
        return widget

    def _apply_language(self, language: str, persist: bool = True):
        if language not in LANGUAGE_CODES:
            language = "zh"
        self._gui_language = language
        self.root.title(self._tr("window_title"))
        for key, widget in self._translatable_widgets.items():
            widget.configure(text=self._tr(key))
        self.audio_entry.set_browse_text(self._tr("browse"))
        self.subtitle_entry.set_browse_text(self._tr("browse"))
        self.model_entry.set_browse_text(self._tr("browse"))
        self.cudnn_entry.set_browse_text(self._tr("browse"))
        dialog_texts = {
            key: self._tr(key)
            for key in (
                "select_folder",
                "select_file",
                "media_files",
                "subtitle_files",
                "all_files",
            )
        }
        for entry in (
            self.audio_entry,
            self.subtitle_entry,
            self.model_entry,
            self.cudnn_entry,
        ):
            entry.set_dialog_texts(dialog_texts)
        self.gui_language_var.set(LANGUAGE_LABELS[language])
        if persist:
            self.config.set("GUI", "language", language)
            self.config.save()

    def _on_gui_language_change(self, _event=None):
        label_to_code = {label: code for code, label in LANGUAGE_LABELS.items()}
        language = label_to_code.get(self.gui_language_var.get(), "zh")
        self._apply_language(language)
    
    def _set_app_icon(self):
        """如果存在图标文件则设置应用程序图标。"""
        icon_path = Path(__file__).resolve().parent / "app/asset" / "openai.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception as e:
                print(f"设置图标失败: {e}")

    # ==================================================================
    # 界面构建
    # ==================================================================

    def _build_ui(self):
        """组装完整的界面布局。"""
        #
        # ── 输入 ──────────────────────────────────────────────
        #
        input_frame = self._register_text_widget(
            "input", ttk.LabelFrame(self.root, text="输入", padding=10)
        )
        input_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        label_keys = ["audio_file", "subtitle_file"]
        modes = ["file", "file"]
        entries: list[DragDropEntry] = []

        for i, (label_key, mode) in enumerate(zip(label_keys, modes)):
            label = self._register_text_widget(
                label_key,
                ttk.Label(input_frame, text=self._tr(label_key), width=14, anchor=tk.E),
            )
            label.grid(
                row=i, column=0, sticky=tk.W, pady=3
            )
            entry = DragDropEntry(input_frame, mode=mode)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=3)
            entries.append(entry)

        input_frame.columnconfigure(1, weight=1)

        # 便捷别名
        self.audio_entry = entries[0]
        self.subtitle_entry = entries[1]

        # 两个选项放在同一个横向容器中，紧挨显示，避免被 grid 的可伸缩列拉开。
        options_frame = ttk.Frame(input_frame)
        options_frame.grid(row=2, column=1, sticky=tk.W, pady=3)

        # 允许字幕与音频文件名不同（取消勾选时会做文件名匹配检查）
        self.allow_subtitle_mismatch_var = tk.BooleanVar()
        self.allow_subtitle_mismatch_check = self._register_text_widget(
            "allow_subtitle_mismatch",
            ttk.Checkbutton(
                options_frame,
                text=self._tr("allow_subtitle_mismatch"),
                variable=self.allow_subtitle_mismatch_var,
            ),
        )
        self.allow_subtitle_mismatch_check.pack(side=tk.LEFT)

        # VAD 和 ASR 生成同名字幕时，允许继续并由后续输出覆盖文件
        self.overwrite_if_exists_var = tk.BooleanVar()
        self.overwrite_if_exists_check = self._register_text_widget(
            "overwrite_if_exists",
            ttk.Checkbutton(
                options_frame,
                text=self._tr("overwrite_if_exists"),
                variable=self.overwrite_if_exists_var,
            ),
        )
        self.overwrite_if_exists_check.pack(side=tk.LEFT, padx=(12, 0))

        #
        # ── 参数（VAD | Whisper）──────────────────────────────
        #
        params_frame = ttk.Frame(self.root)
        params_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self._build_vad_frame(params_frame)
        self._build_whisper_frame(params_frame)

        #
        # ── 控制 + 高级（并排）───────────────────────────────
        #
        tray_frame = ttk.Frame(self.root)
        tray_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # -- 控制（左侧，可拉伸）--
        controls_frame = self._register_text_widget(
            "controls", ttk.LabelFrame(tray_frame, text="控制", padding=10)
        )
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X)

        self.save_btn = self._register_text_widget(
            "save_config",
            ttk.Button(
                btn_frame, text=self._tr("save_config"), command=self._save_config, width=12
            ),
        )
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.vad_btn = self._register_text_widget(
            "run_vad",
            ttk.Button(
                btn_frame, text=self._tr("run_vad"), command=self._run_vad, width=12
            ),
        )
        self.vad_btn.pack(side=tk.LEFT, padx=5)

        self.asr_btn = self._register_text_widget(
            "run_asr",
            ttk.Button(
                btn_frame, text=self._tr("run_asr"), command=self._run_asr, width=12
            ),
        )
        self.asr_btn.pack(side=tk.LEFT, padx=5)

        self.run_all_btn = self._register_text_widget(
            "run_all",
            ttk.Button(
                btn_frame, text=self._tr("run_all"), command=self._run_all, width=12
            ),
        )
        self.run_all_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = self._register_text_widget(
            "cancel",
            ttk.Button(
                btn_frame, text=self._tr("cancel"), command=self._cancel_task, width=8,
                state=tk.DISABLED,
            ),
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = self._register_text_widget(
            "ready", ttk.Label(btn_frame, text=self._tr("ready"), foreground="gray")
        )
        self.status_label.pack(side=tk.RIGHT)

        self.progress = ttk.Progressbar(controls_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(10, 0))

        # -- 高级（右侧，紧凑）--
        advanced_frame = self._register_text_widget(
            "advanced", ttk.LabelFrame(tray_frame, text="高级", padding=6)
        )
        advanced_frame.pack(side=tk.RIGHT, fill=tk.X, padx=(5, 0))

        adv_labels = [("model_dir", "模型目录："), ("cudnn_dir", "cuDNN 目录：")]
        adv_modes = ["folder", "folder"]
        adv_entries: list[DragDropEntry] = []

        for i, ((label_key, label_text), mode) in enumerate(zip(adv_labels, adv_modes)):
            label = self._register_text_widget(
                label_key,
                ttk.Label(advanced_frame, text=label_text, width=14, anchor=tk.E),
            )
            label.grid(
                row=i, column=0, sticky=tk.W, pady=3
            )
            entry = DragDropEntry(advanced_frame, mode=mode)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=3)
            adv_entries.append(entry)

        advanced_frame.columnconfigure(1, weight=1)

        self.model_entry = adv_entries[0]
        self.cudnn_entry = adv_entries[1]

        self.gui_language_var = tk.StringVar(value=LANGUAGE_LABELS["zh"])
        self.gui_language_label = self._register_text_widget(
            "gui_language",
            ttk.Label(advanced_frame, text=self._tr("gui_language"), anchor=tk.E),
        )
        self.gui_language_label.grid(row=2, column=0, sticky=tk.W, pady=3)
        self.gui_language_combo = ttk.Combobox(
            advanced_frame,
            textvariable=self.gui_language_var,
            values=[LANGUAGE_LABELS["zh"], LANGUAGE_LABELS["en"]],
            state="readonly",
            width=14,
        )
        self.gui_language_combo.grid(row=2, column=1, sticky=tk.W, pady=3)
        self.gui_language_combo.bind("<<ComboboxSelected>>", self._on_gui_language_change)

        #
        # ── 日志 ──────────────────────────────────────────────
        #
        log_frame = self._register_text_widget(
            "log", ttk.LabelFrame(self.root, text="日志", padding=10)
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, state=tk.DISABLED, height=14, font=("Microsoft YaHei UI", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def _build_vad_frame(self, parent: ttk.Frame):
        """VAD 参数子框架。"""
        frame = self._register_text_widget(
            "vad_parameters",
            ttk.LabelFrame(parent, text=self._tr("vad_parameters"), padding=10),
        )
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # start_threshold
        self._register_text_widget(
            "start_threshold",
            ttk.Label(frame, text=self._tr("start_threshold"), anchor=tk.W),
        ).grid(
            row=0, column=0, sticky=tk.W, pady=1
        )
        self.start_threshold_var = tk.DoubleVar()
        scale = ttk.Scale(
            frame, from_=0.0, to=1.0, variable=self.start_threshold_var,
            orient=tk.HORIZONTAL, command=self._on_start_threshold_change,
        )
        scale.grid(row=0, column=1, sticky=tk.EW, padx=(5, 5))
        self.start_threshold_label = ttk.Label(frame, text="0.60", width=5, anchor=tk.E)
        self.start_threshold_label.grid(row=0, column=2, sticky=tk.E)

        # end_threshold
        self._register_text_widget(
            "end_threshold",
            ttk.Label(frame, text=self._tr("end_threshold"), anchor=tk.W),
        ).grid(
            row=1, column=0, sticky=tk.W, pady=1
        )
        self.end_threshold_var = tk.DoubleVar()
        scale2 = ttk.Scale(
            frame, from_=0.0, to=1.0, variable=self.end_threshold_var,
            orient=tk.HORIZONTAL, command=self._on_end_threshold_change,
        )
        scale2.grid(row=1, column=1, sticky=tk.EW, padx=(5, 5))
        self.end_threshold_label = ttk.Label(frame, text="0.30", width=5, anchor=tk.E)
        self.end_threshold_label.grid(row=1, column=2, sticky=tk.E)

        # spinboxes
        vad_fields = [
            ("min_speech", "min_speech_var", 0, 60000),
            ("max_speech", "max_speech_var", 0, 120000),
            ("merge_silence", "merge_silence_var", 0, 5000),
            ("extend_speech", "extend_speech_var", 0, 2000),
        ]
        for i, (label_key, attr, frm, to) in enumerate(vad_fields, start=2):
            self._register_text_widget(
                label_key,
                ttk.Label(frame, text=self._tr(label_key), anchor=tk.W),
            ).grid(
                row=i, column=0, sticky=tk.W, pady=1
            )
            var = tk.IntVar()
            setattr(self, attr, var)
            spin = ttk.Spinbox(frame, textvariable=var, from_=frm, to=to, width=8)
            spin.grid(row=i, column=1, sticky=tk.W, padx=5)

        frame.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def _build_whisper_frame(self, parent: ttk.Frame):
        """Whisper 参数子框架。"""
        frame = self._register_text_widget(
            "whisper_parameters",
            ttk.LabelFrame(parent, text=self._tr("whisper_parameters"), padding=10),
        )
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # language
        self._register_text_widget(
            "language",
            ttk.Label(frame, text=self._tr("language"), anchor=tk.W),
        ).grid(
            row=0, column=0, sticky=tk.W, pady=1
        )
        self.language_var = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.language_var, width=14)
        combo["values"] = [
            "auto", "ja", "zh", "en", "ko", "fr", "de", "es", "pt", "ru", "ar", "th", "vi",
        ]
        combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=1)

        # initial_prompt
        self._register_text_widget(
            "initial_prompt",
            ttk.Label(frame, text=self._tr("initial_prompt"), anchor=tk.W),
        ).grid(
            row=1, column=0, sticky=tk.W, pady=1
        )
        self.initial_prompt_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.initial_prompt_var, width=20).grid(
            row=1, column=1, sticky=tk.EW, padx=5, pady=1
        )

        # hotwords
        self._register_text_widget(
            "hotwords",
            ttk.Label(frame, text=self._tr("hotwords"), anchor=tk.W),
        ).grid(
            row=2, column=0, sticky=tk.W, pady=1
        )
        self.hotwords_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.hotwords_var, width=20).grid(
            row=2, column=1, sticky=tk.EW, padx=5, pady=1
        )

        # beam_size
        self._register_text_widget(
            "beam_size",
            ttk.Label(frame, text=self._tr("beam_size"), anchor=tk.W),
        ).grid(
            row=3, column=0, sticky=tk.W, pady=1
        )
        self.beam_size_var = tk.IntVar()
        ttk.Spinbox(frame, textvariable=self.beam_size_var, from_=1, to=10, width=8).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=1
        )

        # best_of
        self._register_text_widget(
            "best_of",
            ttk.Label(frame, text=self._tr("best_of"), anchor=tk.W),
        ).grid(
            row=4, column=0, sticky=tk.W, pady=1
        )
        self.best_of_var = tk.IntVar()
        ttk.Spinbox(frame, textvariable=self.best_of_var, from_=1, to=10, width=8).grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=1
        )

        # checkboxes
        self.word_timestamps_var = tk.BooleanVar()
        self._register_text_widget(
            "word_timestamps",
            ttk.Checkbutton(
                frame, text=self._tr("word_timestamps"), variable=self.word_timestamps_var
            ),
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=1)

        self.log_progress_var = tk.BooleanVar()
        self._register_text_widget(
            "log_progress",
            ttk.Checkbutton(
                frame, text=self._tr("log_progress"), variable=self.log_progress_var
            ),
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=1)

        frame.columnconfigure(1, weight=1)

    # ==================================================================
    # 配置 ↔ 界面绑定
    # ==================================================================

    def _load_config_to_ui(self):
        """从配置对象填充所有界面字段。"""
        self.config.load()

        gui_language = self.config.get("GUI", "language", "zh").lower()
        if gui_language not in LANGUAGE_CODES:
            gui_language = "zh"
        self._apply_language(gui_language, persist=False)

        # 路径
        self.model_entry.set(self.config.get("PATHS", "model_dir"))
        self.audio_entry.set(self.config.get("PATHS", "audio_path"))
        self.subtitle_entry.set(self.config.get("PATHS", "subtitle_path"))
        self.cudnn_entry.set(self.config.get("PATHS", "cudnn_dir"))

        # VAD
        self.start_threshold_var.set(self.config.get_float("VAD", "start_threshold", 0.6))
        self.end_threshold_var.set(self.config.get_float("VAD", "end_threshold", 0.3))
        self.min_speech_var.set(self.config.get_int("VAD", "min_speech_ms", 2000))
        self.max_speech_var.set(self.config.get_int("VAD", "max_speech_ms", 30000))
        self.merge_silence_var.set(self.config.get_int("VAD", "merge_silence_ms", 350))
        self.extend_speech_var.set(self.config.get_int("VAD", "extend_speech_ms", 50))

        # Whisper
        self.language_var.set(self.config.get("WHISPER", "language", "ja"))
        self.initial_prompt_var.set(self.config.get("WHISPER", "initial_prompt"))
        self.hotwords_var.set(self.config.get("WHISPER", "hotwords"))
        self.beam_size_var.set(self.config.get_int("WHISPER", "beam_size", 5))
        self.best_of_var.set(self.config.get_int("WHISPER", "best_of", 3))
        self.word_timestamps_var.set(self.config.get_bool("WHISPER", "word_timestamps"))
        self.log_progress_var.set(self.config.get_bool("WHISPER", "log_progress"))
        self.allow_subtitle_mismatch_var.set(
            self.config.get_bool("WHISPER", "allow_subtitle_mismatch")
        )
        self.overwrite_if_exists_var.set(
            self.config.get_bool("WHISPER", "overwrite_if_exists")
        )

    def _save_ui_to_config(self):
        """将界面上的所有值写入配置对象（不涉及文件 I/O）。"""
        # 路径
        self.config.set("PATHS", "model_dir", self.model_entry.get())
        self.config.set("PATHS", "audio_path", self.audio_entry.get())
        self.config.set("PATHS", "subtitle_path", self.subtitle_entry.get())
        self.config.set("PATHS", "cudnn_dir", self.cudnn_entry.get())

        # VAD
        self.config.set("VAD", "start_threshold", self.start_threshold_var.get())
        self.config.set("VAD", "end_threshold", self.end_threshold_var.get())
        self.config.set("VAD", "min_speech_ms", self.min_speech_var.get())
        self.config.set("VAD", "max_speech_ms", self.max_speech_var.get())
        self.config.set("VAD", "merge_silence_ms", self.merge_silence_var.get())
        self.config.set("VAD", "extend_speech_ms", self.extend_speech_var.get())

        # Whisper
        self.config.set("WHISPER", "language", self.language_var.get())
        self.config.set("WHISPER", "initial_prompt", self.initial_prompt_var.get())
        self.config.set("WHISPER", "hotwords", self.hotwords_var.get())
        self.config.set("WHISPER", "beam_size", self.beam_size_var.get())
        self.config.set("WHISPER", "best_of", self.best_of_var.get())
        self.config.set(
            "WHISPER", "word_timestamps",
            str(self.word_timestamps_var.get()).lower(),
        )
        self.config.set(
            "WHISPER", "log_progress",
            str(self.log_progress_var.get()).lower(),
        )
        self.config.set(
            "WHISPER", "allow_subtitle_mismatch",
            str(self.allow_subtitle_mismatch_var.get()).lower(),
        )
        self.config.set(
            "WHISPER", "overwrite_if_exists",
            str(self.overwrite_if_exists_var.get()).lower(),
        )
        self.config.set("GUI", "language", self._gui_language)

    def _save_config(self):
        self._save_ui_to_config()
        self.config.save()
        self._log(self._tr("config_saved"))

    # ==================================================================
    # 滑块回调
    # ==================================================================

    def _on_start_threshold_change(self, value):
        self.start_threshold_label.configure(text=f"{float(value):.2f}")

    def _on_end_threshold_change(self, value):
        self.end_threshold_label.configure(text=f"{float(value):.2f}")

    # ==================================================================
    # 日志
    # ==================================================================

    def _log(self, msg: str):
        """线程安全：将消息推入日志队列。"""
        self.log_queue.put(msg)

    def _poll_log(self):
        """定期将日志队列中的消息写入文本框。"""
        changed = False
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            # 普通日志到达：若存在进度行，先固化到日志历史，再追加新日志
            if self._progress_text is not None:
                self._log_lines.append(self._progress_text)
                self._progress_text = None
            self._log_lines.append(msg)
            changed = True

        while not self.progress_queue.empty():
            try:
                msg = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            # 新进度覆盖旧进度（进度行始终是最后一行，原地刷新不刷屏）
            self._progress_text = msg
            changed = True

        if changed:
            self._render_log()
        try:
            self.root.after(100, self._poll_log)
        except tk.TclError:
            pass  # 窗口已销毁

    def _render_log(self):
        """用 Python 侧缓存的日志内容整体重建文本框。"""
        lines = list(self._log_lines)
        if self._progress_text is not None:
            lines.append(self._progress_text)
        text = "\n".join(lines)
        try:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", text)
            self.log_text.yview_moveto(1.0)  # 滚动到底部（纯视口操作，不依赖索引算术）
            self.log_text.configure(state=tk.DISABLED)
        except tk.TclError:
            pass

    # ==================================================================
    # 按钮状态管理
    # ==================================================================

    def _set_buttons_state(self, state: str):
        for btn in (self.vad_btn, self.asr_btn, self.run_all_btn, self.save_btn):
            btn.configure(state=state)
        self.gui_language_combo.configure(
            state="disabled" if state == tk.DISABLED else "readonly"
        )

    def _set_status(self, text: str, color: str = "gray"):
        self.status_label.configure(text=text, foreground=color)

    def _clear_entry_focus(self, event):
        """点击输入控件以外的区域时清除焦点，停止光标闪烁。"""
        try:
            widget_class = event.widget.winfo_class()
        except AttributeError:
            # event.widget 可能是原始 Tk 路径字符串（例如来自
            # Combobox 下拉列表框），而非控件对象。
            return

        keep_focus_classes = {
            "Entry",
            "TEntry",
            "Text",
            "Spinbox",
            "TSpinbox",
            "TCombobox",
        }

        if widget_class in keep_focus_classes:
            return

        self.root.focus_set()

    # ==================================================================
    # 校验
    # ==================================================================

    @staticmethod
    def _generated_vad_asr_paths(
        audio_path: str, subtitle_path: Optional[str] = None
    ) -> tuple[str, str] | None:
        """计算 VAD 与 ASR 的目标字幕路径，用于执行前的同名保护。"""
        if not audio_path:
            return None

        audio_p = Path(audio_path)
        vad_path = audio_p.parent / f"{audio_p.stem}_vad.srt"
        subtitle_p = Path(subtitle_path or audio_p.stem)
        asr_path = audio_p.parent / f"{subtitle_p.stem}_asr.srt"
        return str(vad_path), str(asr_path)

    @staticmethod
    def _same_filename(vad_path: str, asr_path: str) -> bool:
        """按 Windows 文件名规则判断两个输出文件名是否相同。"""
        return Path(vad_path).name.casefold() == Path(asr_path).name.casefold()

    def _check_same_vad_asr_filename(
        self, subtitle_path: Optional[str] = None
    ) -> bool:
        """阻止 VAD/ASR 生成同名字幕，除非用户明确允许强制覆盖。"""
        if self.overwrite_if_exists_var.get():
            return True

        paths = self._generated_vad_asr_paths(
            self.audio_entry.get(),
            subtitle_path or self.subtitle_entry.get() or self.vad_srt_path,
        )
        if paths is None:
            return True

        vad_path, asr_path = paths
        if not self._same_filename(vad_path, asr_path):
            return True

        warning = self._tr(
            "same_vad_asr_filename_warning",
            vad_path=vad_path,
            asr_path=asr_path,
        )
        self._log(f"⚠️ {warning}")
        messagebox.showwarning(self._tr("notice"), warning)
        return False

    def _validate_common(self) -> tuple[list[str], list[str]]:
        """VAD 和 ASR 之前都需要进行的检查。

        Returns:
            (errors, warnings) 元组；errors 需要阻断运行，warnings 仅提示。
        """
        errors: list[str] = []
        warnings: list[str] = []
        audio_path = self.audio_entry.get()
        if not audio_path:
            errors.append(self._tr("audio_required"))
        elif not Path(audio_path).exists():
            errors.append(self._tr("audio_missing", path=audio_path))

        import shutil
        if shutil.which("ffmpeg") is None:
            errors.append(self._tr("ffmpeg_missing"))
        return errors, warnings

    def _validate_vad_params(self) -> tuple[list[str], list[str]]:
        """VAD 参数合理性检查。返回 (errors, warnings)。"""
        errors: list[str] = []
        warnings: list[str] = []

        start_th = self.start_threshold_var.get()
        end_th = self.end_threshold_var.get()
        if start_th < end_th:
            errors.append(
                self._tr("threshold_order", start=start_th, end=end_th)
            )

        min_speech = self.min_speech_var.get()
        max_speech = self.max_speech_var.get()
        if min_speech < 0 or max_speech < 0:
            errors.append(self._tr("vad_duration_negative"))
        elif min_speech > max_speech:
            errors.append(
                self._tr(
                    "vad_duration_order",
                    min_ms=min_speech,
                    max_ms=max_speech,
                )
            )

        return errors, warnings

    def _validate_asr(self) -> tuple[list[str], list[str]]:
        """ASR 之前需要进行的检查。返回 (errors, warnings)。"""
        errors, warnings = self._validate_common()

        model_dir = self.model_entry.get()
        if not model_dir:
            errors.append(self._tr("model_required"))
        elif not Path(model_dir).is_dir():
            errors.append(self._tr("model_missing", path=model_dir))
        else:
            missing_model_files = self._missing_files(
                model_dir, REQUIRED_MODEL_FILES
            )
            if missing_model_files:
                errors.append(
                    self._tr(
                        "model_files_missing",
                        path=model_dir,
                        files=", ".join(missing_model_files),
                    )
                )

        subtitle_path = self.subtitle_entry.get() or self.vad_srt_path
        if not subtitle_path:
            errors.append(self._tr("subtitle_required"))
        elif not Path(subtitle_path).is_file():
            errors.append(self._tr("subtitle_missing", path=subtitle_path))
        else:
            # 字幕与当前音频是否匹配（仅警告，不阻断；勾选“允许不同名”后跳过）
            audio_path = self.audio_entry.get()
            if audio_path and not self.allow_subtitle_mismatch_var.get():
                audio_stem = Path(audio_path).stem
                sub_name = Path(subtitle_path).name
                if not sub_name.startswith(audio_stem):
                    warnings.append(
                        self._tr(
                            "subtitle_mismatch",
                            subtitle=sub_name,
                            audio=audio_stem,
                        )
                    )

        # cuDNN 目录已填写但缺少关键 DLL → 警告（运行时会回退到 CPU）。
        # 不填写目录仍表示用户主动选择 CPU，因此不提示运行库错误。
        cudnn_dir = self.cudnn_entry.get()
        if cudnn_dir:
            missing_cuda_dlls = self._missing_files(
                cudnn_dir, REQUIRED_CUDA_DLLS
            )
            if missing_cuda_dlls:
                warnings.append(
                    self._tr(
                        "cudnn_warning",
                        path=cudnn_dir,
                        files=", ".join(missing_cuda_dlls),
                    )
                )

        return errors, warnings

    def _report_validation(self, errors, warnings) -> bool:
        """汇总校验结果并弹窗提示。

        errors    → 逐条写日志 + 错误弹窗，阻断运行
        warnings  → 逐条写日志 + 警告弹窗，不阻断运行

        Returns:
            True 表示无错误、可以运行；False 表示存在错误、应阻断。
        """
        for e in errors:
            self._log(f"❌ {e}")
        for w in warnings:
            self._log(f"⚠️ {w}")

        if errors:
            messagebox.showerror(self._tr("validation_error"), "\n".join(errors))
            return False
        if warnings:
            messagebox.showwarning(self._tr("notice"), "\n".join(warnings))
        return True

    @staticmethod
    def _missing_files(directory: str, required_files: tuple[str, ...]) -> list[str]:
        """Return required filenames that are absent from *directory*."""
        directory_path = Path(directory)
        return [
            filename
            for filename in required_files
            if not (directory_path / filename).is_file()
        ]

    def _cudnn_is_usable(self, cudnn_dir: str) -> bool:
        """cuDNN 目录是否包含本项目需要的 CUDA 12 DLL 文件。"""
        return not self._missing_files(cudnn_dir, REQUIRED_CUDA_DLLS)

    # ==================================================================
    # 线程管理
    # ==================================================================

    def _post(self, fn):
        """将回调安全地投递到主线程；窗口已销毁时静默忽略。"""
        try:
            self.root.after(0, fn)
        except tk.TclError:
            pass

    def _start_thread(self, target, settings: TaskSettings):
        self.running = True
        self._cancel_event.clear()
        self.progress.start()
        self._set_buttons_state(tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        self._set_status(self._tr("running"), "#0078D4")

        def wrapper():
            try:
                target(settings)
            except TaskCancelledError:
                self._log(self._translate(settings.gui_language, "task_cancelled"))
            except Exception as exc:
                import traceback
                self._log(self._translate(settings.gui_language, "error_log", error=exc))
                self._log(traceback.format_exc())
                err_msg = str(exc) or exc.__class__.__name__
                error_title = self._translate(settings.gui_language, "runtime_error")
                error_details = self._translate(settings.gui_language, "error_details")
                self._post(lambda: messagebox.showerror(
                    error_title,
                    f"{err_msg}\n\n{error_details}",
                ))
            finally:
                self._post(self._task_done)

        threading.Thread(target=wrapper, daemon=True).start()

    def _task_done(self):
        """后台任务结束后在主线程中调用。"""
        self.running = False
        self.progress.stop()
        self.cancel_btn.configure(state=tk.DISABLED)
        self._set_buttons_state(tk.NORMAL)
        self._set_status(self._tr("ready"), "gray")

    def _cancel_task(self):
        """请求取消当前后台任务（ASR 分块循环中生效）。"""
        self._cancel_event.set()
        self.cancel_btn.configure(state=tk.DISABLED)
        self._set_status(self._tr("cancelling"), "#B00020")
        self._log(self._tr("cancelling_log"))

    def _on_close(self):
        """窗口关闭：任务运行中先询问用户。"""
        if self.running:
            if not messagebox.askyesno(
                self._tr("confirm_exit"),
                self._tr("exit_warning"),
            ):
                return
            self._cancel_event.set()
        self.root.destroy()

    # ==================================================================
    # 运行按钮
    # ==================================================================

    def _collect_settings(self) -> TaskSettings:
        """主线程：将界面字段冻结为不可变快照，供后台线程使用。"""
        audio_path = self.audio_entry.get()
        model_dir = self.model_entry.get()
        subtitle_path = self.subtitle_entry.get() or self.vad_srt_path
        cudnn_dir = self.cudnn_entry.get()

        # cuDNN 目录不可用时回退到 CPU（校验阶段已弹窗提示）
        device = "cuda" if self._cudnn_is_usable(cudnn_dir) else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        cudnn_dir = cudnn_dir if device == "cuda" else ""

        language = self.language_var.get().strip()
        if language.lower() in ("auto", ""):
            language = None

        return TaskSettings(
            audio_path=audio_path,
            model_dir=model_dir,
            subtitle_path=subtitle_path,
            cudnn_dir=cudnn_dir,
            device=device,
            compute_type=compute_type,
            start_threshold=float(self.start_threshold_var.get()),
            end_threshold=float(self.end_threshold_var.get()),
            min_speech_ms=int(self.min_speech_var.get()),
            max_speech_ms=int(self.max_speech_var.get()),
            merge_silence_ms=int(self.merge_silence_var.get()),
            extend_speech_ms=int(self.extend_speech_var.get()),
            language=language,
            initial_prompt=self.initial_prompt_var.get().strip() or None,
            hotwords=self.hotwords_var.get().strip() or None,
            beam_size=int(self.beam_size_var.get()),
            best_of=int(self.best_of_var.get()),
            word_timestamps=bool(self.word_timestamps_var.get()),
            log_progress=bool(self.log_progress_var.get()),
            gui_language=self._gui_language,
            overwrite_if_exists=bool(
                self.overwrite_if_exists_var.get()
            ),
        )

    def _run_vad(self):
        errors, warnings = self._validate_common()
        vad_errors, vad_warnings = self._validate_vad_params()
        if not self._report_validation(errors + vad_errors, warnings + vad_warnings):
            return
        if not self._check_same_vad_asr_filename():
            return
        # 校验通过后才保存配置，避免无效输入被写入 gui_config.ini
        self._save_ui_to_config()
        self.config.save()
        self._start_thread(self._vad_thread, self._collect_settings())

    def _run_asr(self):
        errors, warnings = self._validate_asr()
        if not self._report_validation(errors, warnings):
            return
        if not self._check_same_vad_asr_filename():
            return
        # 校验通过后才保存配置，避免无效输入被写入 gui_config.ini
        self._save_ui_to_config()
        self.config.save()
        self._start_thread(self._asr_thread, self._collect_settings())

    def _run_all(self):
        errors, warnings = self._validate_common()
        vad_errors, vad_warnings = self._validate_vad_params()
        model_dir = self.model_entry.get()
        if not model_dir:
            errors.append(self._tr("model_required"))
        elif not Path(model_dir).is_dir():
            errors.append(self._tr("model_missing", path=model_dir))
        if not self._report_validation(errors + vad_errors, warnings + vad_warnings):
            return
        # 一键运行的 ASR 输入是本次 VAD 生成的字幕，因此按该路径计算冲突。
        audio_path = self.audio_entry.get()
        predicted_vad_path = (
            str(Path(audio_path).parent / f"{Path(audio_path).stem}_vad.srt")
            if audio_path else None
        )
        if not self._check_same_vad_asr_filename(predicted_vad_path):
            return
        # 校验通过后才保存配置，避免无效输入被写入 gui_config.ini
        self._save_ui_to_config()
        self.config.save()
        self._start_thread(self._all_thread, self._collect_settings())

    # ==================================================================
    # VAD 线程（后台）
    # ==================================================================

    def _vad_thread(self, settings: TaskSettings, notify: bool = True) -> Optional[str]:
        """运行 FireRedVAD 并生成 _vad.srt 文件。

        返回生成的 SRT 路径；失败或被取消时返回 None。
        """
        from app.FireRedVAD.main import vad_detect

        if self._cancel_event.is_set():
            self._log(self._translate(settings.gui_language, "vad_cancelled"))
            return None

        vad_srt_path = settings.vad_srt_path
        tmp_srt_path = vad_srt_path + ".tmp"

        if not settings.overwrite_if_exists:
            # 一键运行的后续 ASR 使用本次 VAD 输出，而不是界面中的旧字幕路径。
            asr_input_path = settings.vad_srt_path if not notify else settings.subtitle_path
            output_paths = self._generated_vad_asr_paths(
                settings.audio_path, asr_input_path
            )
            if output_paths and self._same_filename(*output_paths):
                warning = self._translate(
                    settings.gui_language,
                    "same_vad_asr_filename_warning",
                    vad_path=output_paths[0],
                    asr_path=output_paths[1],
                )
                self._log(f"⚠️ {warning}")
                self._post(lambda: messagebox.showwarning(
                    self._translate(settings.gui_language, "notice"),
                    warning,
                ))
                return None

        fireredvad_dir = Path(__file__).resolve().parent / "app" / "FireRedVAD"
        model_onnx = str(fireredvad_dir / "fireredvad_vad.onnx")
        cmvn_ark = str(fireredvad_dir / "cmvn.ark")

        self._log("=" * 50)
        self._log(self._translate(settings.gui_language, "vad_start"))
        self._log(self._translate(
            settings.gui_language,
            "vad_audio",
            path=settings.audio_path,
        ))
        self._log(self._translate(
            settings.gui_language,
            "vad_output",
            path=vad_srt_path,
        ))

        result = vad_detect(
            wav_path=settings.audio_path,
            model_path=model_onnx,
            cmvn_path=cmvn_ark,
            output_srt=tmp_srt_path,
            start_threshold=settings.start_threshold,
            end_threshold=settings.end_threshold,
            min_speech_ms=settings.min_speech_ms,
            max_speech_ms=settings.max_speech_ms,
            merge_silence_ms=settings.merge_silence_ms,
            extend_speech_ms=settings.extend_speech_ms,
        )

        timestamps = result.get("timestamps", [])
        dur = result.get("dur", 0.0)

        # 输出原子化：先写临时文件，成功后替换正式文件
        if not os.path.isfile(tmp_srt_path):
            self._log(self._translate(
                settings.gui_language,
                "vad_no_output",
                path=vad_srt_path,
            ))
            return None
        os.replace(tmp_srt_path, vad_srt_path)

        self.vad_srt_path = vad_srt_path

        # 更新界面输入框并持久化配置（主线程回调，避免后台访问 Tk）
        self._post(lambda: self._persist_subtitle_path(vad_srt_path))

        self._log(self._translate(
            settings.gui_language,
            "vad_done",
            count=len(timestamps),
            duration=dur,
        ))
        self._log(self._translate(
            settings.gui_language,
            "srt_saved",
            path=vad_srt_path,
        ))
        self._log("─" * 50)
        if notify:
            self._log(self._translate(settings.gui_language, "vad_review"))
            complete_title = self._translate(settings.gui_language, "vad_complete")
            complete_message = self._translate(
                settings.gui_language,
                "vad_complete_message",
                count=len(timestamps),
                duration=dur,
                path=vad_srt_path,
            )
            self._post(lambda: messagebox.showinfo(
                complete_title,
                complete_message,
            ))
        return vad_srt_path

    # ==================================================================
    # 模型缓存
    # ==================================================================

    def _get_or_load_model(self, model_dir: str, device: str, compute_type: str,
                           cudnn_dir: str, gui_language: str):
        """返回缓存的 WhisperSubtitle 模型；若模型目录/设备/计算类型
        与上次加载不同，则重新加载新模型。"""
        key = (model_dir, device, compute_type)
        if self._model is not None and self._model_key == key:
            self._log(self._translate(gui_language, "model_reuse"))
            return self._model

        if self._model is not None:
            self._log(self._translate(gui_language, "model_reload"))
        else:
            self._log(self._translate(gui_language, "model_first_load"))

        if cudnn_dir:
            load_cudnn(cudnn_dir=cudnn_dir)

        self._model = WhisperSubtitle(
            model_dir,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
        self._model_key = key
        return self._model

    # ==================================================================
    # ASR 线程（后台）
    # ==================================================================

    def _asr_thread(self, settings: TaskSettings, subtitle_path: Optional[str] = None):
        """使用字幕分块运行 WhisperSubtitle 转写。"""
        if self._cancel_event.is_set():
            self._log(self._translate(settings.gui_language, "asr_cancelled"))
            return

        # 一键运行会显式传入 VAD 生成的字幕路径，避免界面异步更新竞态
        subtitle_path = subtitle_path or settings.subtitle_path
        audio_path = settings.audio_path
        model_dir = settings.model_dir
        asr_srt_path = settings.asr_srt_path(subtitle_path)
        tmp_srt_path = asr_srt_path + ".tmp"

        if not settings.overwrite_if_exists and self._same_filename(
            settings.vad_srt_path, asr_srt_path
        ):
            warning = self._translate(
                settings.gui_language,
                "same_vad_asr_filename_warning",
                vad_path=settings.vad_srt_path,
                asr_path=asr_srt_path,
            )
            self._log(f"⚠️ {warning}")
            self._post(lambda: messagebox.showwarning(
                self._translate(settings.gui_language, "notice"),
                warning,
            ))
            return

        self._log("=" * 50)
        self._log(self._translate(settings.gui_language, "asr_start"))
        self._log(self._translate(settings.gui_language, "asr_model", path=model_dir))
        self._log(self._translate(settings.gui_language, "asr_audio", path=audio_path))
        self._log(self._translate(settings.gui_language, "asr_chunks", path=subtitle_path))
        self._log(self._translate(
            settings.gui_language,
            "asr_device",
            device=settings.device,
            compute_type=settings.compute_type,
        ))
        self._log(self._translate(
            settings.gui_language,
            "asr_language",
            language=settings.language or "auto",
        ))

        self._log(self._translate(settings.gui_language, "model_loading"))
        model = self._get_or_load_model(
            model_dir,
            settings.device,
            settings.compute_type,
            settings.cudnn_dir,
            settings.gui_language,
        )

        self._log(self._translate(settings.gui_language, "transcribing"))

        def on_progress(done: int, total: int):
            self.progress_queue.put(self._translate(
                settings.gui_language,
                "progress",
                done=done,
                total=total,
            ))

        try:
            segments, _ = model.transcribe_with_subtitle_chunks(
                audio=audio_path,
                subtitle_path=subtitle_path,
                beam_size=settings.beam_size,
                best_of=settings.best_of,
                initial_prompt=settings.initial_prompt,
                language=settings.language,
                word_timestamps=settings.word_timestamps,
                log_progress=settings.log_progress,
                hotwords=settings.hotwords,
                progress_callback=on_progress if settings.log_progress else None,
                cancel_event=self._cancel_event,
            )
            segment_list = list(segments)
        except TaskCancelledError:
            self._cleanup_tmp(tmp_srt_path)
            self._log(self._translate(
                settings.gui_language,
                "asr_cancelled_output",
            ))
            raise

        timestamps = [(s.start, s.end) for s in segment_list]
        texts = [s.text if hasattr(s, "text") else "" for s in segment_list]

        # 输出原子化：先写临时文件，成功后替换正式文件
        write_srt_file(path=tmp_srt_path, timestamps=timestamps, texts=texts)
        os.replace(tmp_srt_path, asr_srt_path)

        self._log(self._translate(
            settings.gui_language,
            "asr_done",
            count=len(segment_list),
        ))
        self._log(self._translate(
            settings.gui_language,
            "srt_saved",
            path=asr_srt_path,
        ))
        self._log("=" * 50)

        complete_title = self._translate(settings.gui_language, "asr_complete")
        complete_message = self._translate(
            settings.gui_language,
            "asr_complete_message",
            count=len(segment_list),
            path=asr_srt_path,
        )
        self._post(lambda: messagebox.showinfo(
            complete_title,
            complete_message,
        ))

    # ==================================================================
    # 一键运行线程（后台）
    # ==================================================================

    def _all_thread(self, settings: TaskSettings):
        """一键运行：依次执行 VAD，然后用 VAD 生成的字幕执行 ASR。"""
        vad_srt_path = self._vad_thread(settings, notify=False)
        if self._cancel_event.is_set():
            self._log(self._translate(settings.gui_language, "skip_asr"))
            return
        if vad_srt_path is None:
            self._log(self._translate(settings.gui_language, "no_subtitle"))
            return
        self._asr_thread(settings, subtitle_path=vad_srt_path)

    # ------------------------------------------------------------------
    def _persist_subtitle_path(self, vad_srt_path: str):
        """主线程：将 VAD 生成的字幕路径写入界面输入框与配置文件。"""
        self.subtitle_entry.set(vad_srt_path)
        self.config.set("PATHS", "subtitle_path", vad_srt_path)
        self.config.save()

    def _cleanup_tmp(self, tmp_path: str):
        """删除未完成的临时输出文件。"""
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ============================================================================
# 入口点
# ============================================================================

def main(config_path: Optional[Path] = None):
    """启动图形界面。

    Args:
        config_path: 覆盖 gui_config.ini 的路径。
    """
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = App(root)

    if config_path:
        app.config_path = Path(config_path)
        app.config = ConfigManager(app.config_path)
        app.config.load()
        app._load_config_to_ui()

    root.mainloop()


if __name__ == "__main__":
    main()
