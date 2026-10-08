"""Small filesystem / JSON helpers. / 文件系统与 JSON 小工具。"""

from __future__ import annotations

import json
import os
import shutil
import time


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    """Wall-clock ``HH:MM:SS`` stamp for progress lines."""
    return time.strftime("%H:%M:%S")


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``12.3s`` / ``4m 05.0s`` / ``1h 02m 03s``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {sec:04.1f}s"


def progress(zh: str, en: str, elapsed: float | None = None) -> None:
    """Print one bilingual progress line, optionally with an elapsed time.

    打印一行中英双语进度，可选附带耗时。
    """
    stamp = timestamp()
    message = f"[{stamp}] {zh} / {en}"
    if elapsed is not None:
        message += f" ({format_duration(elapsed)})"
    print(message, flush=True)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=4)


def clear_folder(path: str) -> None:
    """Remove all files/sub-folders inside ``path``. / 清空文件夹。"""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        target = os.path.join(path, name)
        if os.path.isfile(target):
            os.remove(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
