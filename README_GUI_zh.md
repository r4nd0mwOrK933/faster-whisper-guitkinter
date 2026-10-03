# Faster Whisper 图形界面

一个用于本地语音转文字和字幕生成的 Windows 桌面图形界面，基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 构建。

## 功能

- 中文和英文 Tkinter 界面
- 转录前使用 FireRedVAD 进行语音活动检测
- 支持本地 CTranslate2 Whisper 模型
- 生成 SRT 字幕
- 支持 CPU 和 Windows DirectML 执行路径
- 提供用于构建便携式 Windows 文件夹的 PyInstaller 配置

## 环境要求

- Windows 10 或更高版本
- Python 3.14
- 已添加到 `PATH` 的 FFmpeg
- 已在本地下载的 CTranslate2 格式 Whisper 模型

应用程序会调用 `ffmpeg` 命令行程序解码音频。本项目不包含 FFmpeg；请单独安装可再分发的 FFmpeg 构建版本，并遵守其许可条款。

## 从源代码安装和运行

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
# 调试启动：保留终端，便于查看错误输出
python whisper_tkgui.py
```

日常使用源码启动时，可以使用 `pythonw.exe` 隐藏控制台窗口：

```powershell
pythonw.exe whisper_tkgui.py
```

打包后的 Windows GUI 应用也采用无控制台模式，直接运行
`dist\whisper_tkgui\whisper_tkgui.exe` 即可。

首次运行时，程序会根据 gui_config.example.ini 生成 `gui_config.ini`。请填写 Whisper 模型和音频文件的本地路径。不要提交个人路径或凭据。

GUI 需要一个本地 CTranslate2 模型目录。它不会自动下载主要的 Whisper 模型。可以选择从 [faster-whisper 模型集合](https://huggingface.co/Systran) 转换或下载的模型，并指定其本地目录路径。

仓库中包含 FireRedVAD 和 Silero VAD 的运行时资源。主要的 Whisper 模型由于文件较大且有各自的分发条款，特意不包含在仓库中。

## 构建 Windows 应用程序

请在干净的 Python 环境中，从仓库根目录运行以下命令：

```powershell
python -m pip install ".[build]"
python -m PyInstaller --clean --noconfirm build.spec
```

输出目录为 `dist/whisper_tkgui/`。这是一个基于文件夹的应用程序，应作为 ZIP 压缩包分发。输出中包含 `LICENSE`、`THIRD_PARTY_NOTICES` 以及 `licenses/` 下的相关文件。FFmpeg 仍然是单独的 Windows 前置依赖项。

## GitHub Actions

- 可以在 Actions 页面手动启动 `Windows GUI Build`。
- 生成的 Windows ZIP 文件会作为工作流构件上传，供下载。
- 此工作流不会在推送、拉取请求或标签事件时自动运行，也不会创建 GitHub Releases。

该工作流不会下载或打包主要的 Whisper 模型。

## 许可证和致谢

本项目是一个衍生应用程序，其中包含来自 faster-whisper 的代码。第三方许可证、模型来源、校验和以及署名要求请参阅 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。
