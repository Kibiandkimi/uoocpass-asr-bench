#!/usr/bin/env python3
"""下载各后端模型到 ``models/``。

为什么是 Python 而不是 bash+curl
--------------------------------
上一版用 bash + ``curl`` 拼，在 GitHub runner 上**立即失败且没有任何输出**：
``bash -x`` 的跟踪都没落到日志里，而同一个 runner 直接敲 ``curl`` 下载 190MB
却完全正常（见 ``NOTES.md``）。与其继续跟 shell 变量展开和 ``set -e`` 斗，
不如换成 Python：异常有类型、有堆栈、有 URL，重试与多镜像也更好写。

镜像
----
国内网络常常连不上 huggingface.co。这里按顺序试：

* HuggingFace 官方
* ``hf-mirror.com``（镜像，路径规则与官方一致）
* ModelScope（部分模型有镜像）

哪个先成功用哪个，全部失败才报错，并把每个 URL 的失败原因都打出来。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) uoocpass-asr-bench/1.0"

#: HuggingFace 的镜像前缀（顺序即优先级）
HF_HOSTS = (
    "https://huggingface.co",
    "https://hf-mirror.com",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path, *, retries: int = 3, timeout: float = 1800.0) -> bool:
    """带重试的单文件下载。成功返回 True。"""
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"  ✓ 已有 {dest.name}（{dest.stat().st_size/1024/1024:.1f} MB）")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                with part.open("wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                if total and got < total * 0.99:
                    raise OSError(f"下载不完整：{got}/{total} 字节")
            part.replace(dest)
            log(f"  ✓ {dest.name}（{got/1024/1024:.1f} MB）← {url}")
            return True
        except Exception as exc:  # noqa: BLE001 - 每种失败都要能看到
            part.unlink(missing_ok=True)
            log(f"  ! 第 {attempt}/{retries} 次失败 {url}\n      {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return False


def download_mirrored(path: str, dest: Path, **kw) -> bool:
    """按 :data:`HF_HOSTS` 依次试同一个仓库内路径。"""
    for host in HF_HOSTS:
        if download(f"{host}/{path}", dest, **kw):
            return True
    return False


def hf_list(repo: str) -> list[str]:
    """列出仓库文件（走镜像，哪个通用哪个）。"""
    import json

    for host in HF_HOSTS:
        url = f"{host}/api/models/{repo}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return [f["rfilename"] for f in data.get("siblings", [])]
        except Exception as exc:  # noqa: BLE001
            log(f"  ! 列目录失败 {url}：{type(exc).__name__}: {exc}")
    return []


def pick_ggml(files: list[str], base: str) -> str | None:
    """从文件列表里挑一个量化变体：q5_0 > q5_1 > q8_0 > 未量化。"""
    for suffix in ("-q5_0.bin", "-q5_1.bin", "-q8_0.bin", ".bin"):
        for f in files:
            if f == base + suffix or f.endswith("/" + base + suffix):
                return f
    return None


def untar(archive: Path, dest_dir: Path, label: str) -> None:
    """解压 .tar.bz2 / .tar.gz，成功后删掉压缩包省空间。"""
    try:
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
        log(f"  ✓ 解压 {label}")
        archive.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  ✗ 解压失败 {archive.name}：{type(exc).__name__}: {exc}")


def unzip(archive: Path, dest_dir: Path, label: str) -> None:
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
        log(f"  ✓ 解压 {label}")
        archive.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  ✗ 解压失败 {archive.name}：{type(exc).__name__}: {exc}")


#: sherpa-onnx 的模型都发在同一个 GitHub release 下
SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


def fetch_vosk() -> tuple[int, int]:
    ok = bad = 0
    log("== vosk 中文 ==")
    for name, zipname, dirname in (
        ("vosk-model-small-cn-0.22", "vosk-small-cn.zip", "vosk-model-small-cn-0.22"),
        ("vosk-model-cn-0.22", "vosk-big-cn.zip", "vosk-model-cn-0.22"),
    ):
        target = MODELS / dirname
        if target.is_dir():
            log(f"  ✓ 已有 {dirname}")
            ok += 1
            continue
        url = f"https://alphacephei.com/vosk/models/{name}.zip"
        if not download(url, MODELS / zipname):
            bad += 1
            continue
        unzip(MODELS / zipname, MODELS, dirname)
        if target.is_dir():
            ok += 1
        else:
            bad += 1
    return ok, bad


def fetch_whisper_cpp() -> tuple[int, int]:
    ok = bad = 0
    log("== whisper.cpp ggml ==")
    files = None
    for size in ("tiny", "base", "small"):
        base = f"ggml-{size}"
        target = MODELS / f"{base}-q5_0.bin"
        if target.is_file():
            log(f"  ✓ 已有 {target.name}")
            ok += 1
            continue
        if files is None:
            files = hf_list("ggerganov/whisper.cpp")
            log(f"  · HF 仓库可见 {len(files)} 个文件")
        picked = pick_ggml(files, base) if files else None
        if picked:
            log(f"  · {base} → {picked}")
            if download_mirrored(f"ggerganov/whisper.cpp/resolve/main/{picked}",
                                 MODELS / Path(picked).name):
                if Path(picked).name != target.name:
                    target.unlink(missing_ok=True)
                    target.symlink_to(Path(picked).name)
                ok += 1
                continue
        else:
            log(f"  · {base}：没在仓库里找到可用量化文件")
        bad += 1
    return ok, bad


def fetch_sherpa() -> tuple[int, int]:
    """sherpa-onnx 的模型（paraformer / sensevoice / zipformer-ctc / fire-red）。"""
    ok = bad = 0
    #: (显示名, 目录名, 压缩包名)
    want = (
        ("paraformer-zh-2023", "sherpa-onnx-paraformer-zh-2023-09-14",
         "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"),
        ("paraformer-zh-2024", "sherpa-onnx-paraformer-zh-2024-03-09",
         "sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"),
        ("sensevoice", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
         "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"),
        ("zipformer-ctc-zh", "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
         "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2"),
    )
    log("== sherpa-onnx 模型 ==")
    for label, dirname, archive in want:
        target = MODELS / dirname
        if target.is_dir():
            log(f"  ✓ 已有 {dirname}")
            ok += 1
            continue
        if download(f"{SHERPA_RELEASE}/{archive}", MODELS / archive):
            untar(MODELS / archive, MODELS, dirname)
            if target.is_dir():
                ok += 1
            else:
                bad += 1
        else:
            bad += 1
    return ok, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载 ASR 基准用的模型")
    ap.add_argument("--only", help="只下这些（逗号分隔：vosk,whisper.cpp,sherpa）")
    args = ap.parse_args(argv)

    MODELS.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    ok = bad = 0
    for key, fn in (("vosk", fetch_vosk), ("whisper.cpp", fetch_whisper_cpp),
                    ("sherpa", fetch_sherpa)):
        if only and key not in only:
            continue
        a, b = fn()
        ok += a
        bad += b

    log("")
    log("== 小结 ==")
    if shutil.which("du"):
        subprocess.run(["du", "-sh", *sorted(str(p) for p in MODELS.iterdir())][:20])
    log(f"成功 {ok} 项，失败 {bad} 项")
    if bad:
        log("!! 有模型没下到 —— 上面的 `!` 行有每个 URL 的具体失败原因")
        return 1
    log("全部就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
