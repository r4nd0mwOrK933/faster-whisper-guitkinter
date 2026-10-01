# -*- mode: python ; coding: utf-8 -*-
"""
Faster-Whisper PyInstaller 打包配置
------------------------------------
入口: whisper_tkgui.py → GUI 应用
结构: 依赖放入 internal/ 子目录，app/ 保留为可编辑源码
"""

from PyInstaller.utils.hooks import collect_data_files
from os.path import join, basename, dirname, exists
from os import walk, makedirs
from shutil import copyfile, rmtree, copytree

# ==================== 配置选项 ====================

# 可编辑模块（保留为 .py 源码，不编译进 internal/）
EDITABLE_MODULES = ['app', 'app_gui']

# =================================================

block_cipher = None

# -------------------------------------------------------------------
# 1. 收集资源文件 (datas)
# -------------------------------------------------------------------

datas = []

# Silero VAD 模型 (faster_whisper 内置)
datas += [
    ('faster_whisper/assets/silero_vad_v6.onnx', 'faster_whisper/assets'),
    ('faster_whisper/assets/__init__.py', 'faster_whisper/assets'),
]

# FireRedVAD 模型
datas += [
    ('app/FireRedVAD/fireredvad_vad.onnx', 'app/FireRedVAD'),
    ('app/FireRedVAD/cmvn.ark', 'app/FireRedVAD'),
]

# 应用图标
datas += [
    ('app/asset/openai.ico', 'app/asset'),
]

# 配置文件（模板）
# datas += [
#     ('gui_config.example.ini', '.'),
# ]

# 收集 tkinterdnd2
datas += collect_data_files('tkinterdnd2')

# -------------------------------------------------------------------
# 2. 隐藏导入 (hiddenimports)
# -------------------------------------------------------------------

hiddenimports = [
    # GUI
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'tkinterdnd2',

    # Core dependencies
    'numpy',
    'onnxruntime',
    'ctranslate2',
    'tokenizers',
    'av',

    # Utilities
    'tqdm',
    'huggingface_hub',
    'configparser',

    # Image support (Pillow, for icon)
    # 'PIL',
    # 'PIL.Image',

    # Standard library (sometimes missed by analysis)
    'queue',
    'threading',
    'logging',
    'json',
    'subprocess',
    'tempfile',
    'ctypes',
    'glob',
    'dataclasses',
]

# -------------------------------------------------------------------
# 3. 排除模块 (excludes)
# -------------------------------------------------------------------

excludes = [
    # Heavy ML frameworks (not used at runtime)
    'torch',
    'transformers',
    'funasr',
    'pydantic',

    # Alternative GUI frameworks
    'PySide6',
    'PySide2',
    'PyQt5',
    'PyQt6',

    # Visualization / interactive
    'matplotlib',
    'wx',
    'IPython',
    'jupyter',
    'ipykernel',

    # Dev / packaging tools
    'pytest',
    'unittest',
    'setuptools',
    'pip',
    'wheel',
    'black',
    'flake8',
    'isort',

    # Unlikely to be used
    'scipy',
    'pandas',
]

# -------------------------------------------------------------------
# 4. Analysis — 分析主脚本依赖
# -------------------------------------------------------------------

a = Analysis(
    ['whisper_tkgui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=excludes,
    noarchive=True,
)

# -------------------------------------------------------------------
# 5. 过滤系统 CUDA DLL（避免误收集环境中的 CUDA 文件）
# -------------------------------------------------------------------

filtered_binaries = []
for name, src, typ in a.binaries:
    src_lower = src.lower() if isinstance(src, str) else ''

    # 检测是否来自系统 CUDA/NVIDIA 安装目录
    is_system_cuda = (
        '\\nvidia gpu computing toolkit\\cuda\\' in src_lower
        or '\\nvidia\\cudnn\\' in src_lower
        or ('\\cuda\\v' in src_lower and '\\bin\\' in src_lower)
    )

    # 检测不需要的 ONNX Runtime provider DLL
    is_unwanted_onnx = 'onnxruntime_providers_cuda.dll' in name.lower()

    if is_system_cuda or is_unwanted_onnx:
        reason = '环境 CUDA DLL' if is_system_cuda else '冗余 ONNX CUDA DLL'
        print(f'[INFO] 排除 {reason}: {name} (from {src})')
    else:
        filtered_binaries.append((name, src, typ))

a.binaries = filtered_binaries

# -------------------------------------------------------------------
# 6. 过滤可编辑模块 — 从 PyInstaller 输出中移除
#    noarchive=True 时，模块会以 .pyc 形式出现在 a.datas 中；
#    同时模块信息出现在 a.pure 中。
#    将 EDITABLE_MODULES 从两者中移除，使其保持为外部 .py 源文件。
# -------------------------------------------------------------------

for collection_name in ('pure', 'datas'):
    collection = getattr(a, collection_name)
    filtered = []
    for item in collection:
        # item format: (name, src, type)
        mod_name = item[0]
        # 检查是否匹配任何可编辑模块
        is_editable = any(
            mod_name == m
            or mod_name.startswith(m + '/')
            or mod_name.startswith(m + '\\')
            or mod_name.startswith(m + '.')
            for m in EDITABLE_MODULES
        )
        if not is_editable:
            filtered.append(item)
        else:
            print(f'[INFO] 排除可编辑模块 ({collection_name}): {mod_name}')
    setattr(a, collection_name, filtered)

# -------------------------------------------------------------------
# 7. PYZ — 打包纯 Python 模块
# -------------------------------------------------------------------

pyz = PYZ(a.pure)

# -------------------------------------------------------------------
# 8. EXE — 生成可执行文件
# -------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisper_tkgui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app/asset/openai.ico'],
    # 所有第三方依赖放入 internal 子目录
    contents_directory='internal',
)

# -------------------------------------------------------------------
# 9. COLLECT — 收集所有分发文件到 dist/whisper_tkgui/
# -------------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whisper_tkgui',
)

# ===================================================================
# 10. 后处理 — 复制可编辑源码到 dist/whisper_tkgui/ 根目录
# ===================================================================
# PyInstaller 的 Analysis.datas 将这些文件放在 internal/，
# 但可编辑模块需要放在 dist 根目录以便用户直接修改。

dest_root = join('dist', basename(coll.name))  # dist/whisper_tkgui

# --- 10a. 复制可编辑的单个 .py 文件 ---
editable_files = [
    'whisper_tkgui.py',
    'app_gui.py',
    'gui_config.example.ini',
]

for file in editable_files:
    if not exists(file):
        print(f'[WARN] 可编辑文件不存在，跳过: {file}')
        continue
    dest_file = join(dest_root, file)
    dest_folder = dirname(dest_file)
    makedirs(dest_folder, exist_ok=True)
    copyfile(file, dest_file)
    print(f'[INFO] 复制可编辑文件: {file} -> {dest_file}')

# --- 10b. 复制 app/ 包（整个目录，排除 __pycache__） ---
app_src = 'app'
app_dest = join(dest_root, 'app')

if exists(app_dest):
    rmtree(app_dest)

# 手动复制以排除 __pycache__ 和 Exp/
def copy_editable_package(src_dir, dest_dir):
    """Recursively copy a package, excluding __pycache__ and Exp/."""
    for dirpath, dirnames, filenames in walk(src_dir):
        # Exclude __pycache__ and Exp directories
        dirnames[:] = [
            d for d in dirnames
            if d != '__pycache__'
            and d != 'Exp'
            and not d.endswith('.egg-info')
        ]

        for filename in filenames:
            # Only copy .py, .ico, .onnx, .ark (no .pyc)
            if not (filename.endswith('.pyc') or filename.endswith('.pyo')):
                src_file = join(dirpath, filename)
                rel_path = join(
                    dirpath[len(src_dir):].lstrip('\\').lstrip('/'),
                    filename
                )
                dest_file = join(dest_dir, rel_path)
                dest_folder = dirname(dest_file)
                makedirs(dest_folder, exist_ok=True)
                copyfile(src_file, dest_file)
                print(f'[INFO] 复制源文件: {rel_path}')

copy_editable_package(app_src, app_dest)

# --- 10c. 复制许可证和第三方声明 ---
license_files = [
    ('LICENSE', 'LICENSE'),
    ('THIRD_PARTY_NOTICES', 'THIRD_PARTY_NOTICES'),
    ('licenses/FireRedVAD-Apache-2.0.txt', 'licenses/FireRedVAD-Apache-2.0.txt'),
]

for source, relative_dest in license_files:
    if not exists(source):
        print(f'[WARN] 许可证文件不存在，跳过: {source}')
        continue
    dest_file = join(dest_root, relative_dest)
    makedirs(dirname(dest_file), exist_ok=True)
    copyfile(source, dest_file)
    print(f'[INFO] 复制许可证文件: {source} -> {dest_file}')

# --- 10d. 清理 internal/ 中多余的可编辑模块副本 ---
# noarchive=True 时，可编辑模块的 .pyc 可能仍出现在 internal/ 中。
# 从 internal/ 删除它们，避免与根目录的 .py 源文件冲突。
internal_root = join(dest_root, 'internal')
if exists(internal_root):
    for mod_name in EDITABLE_MODULES:
        # 删除 internal/ 中的可编辑模块副本
        mod_path = join(internal_root, mod_name)
        if exists(mod_path):
            rmtree(mod_path)
            print(f'[INFO] 清理 internal/{mod_name}')
        # 也尝试删除 .pyc 单文件
        pyc_path = join(internal_root, mod_name + '.pyc')
        if exists(pyc_path):
            import os as _os
            _os.remove(pyc_path)
            print(f'[INFO] 清理 internal/{mod_name}.pyc')

print()
print('=' * 60)
print('  Faster-Whisper 打包完成！')
print(f'  输出目录: {dest_root}')
print('=' * 60)
