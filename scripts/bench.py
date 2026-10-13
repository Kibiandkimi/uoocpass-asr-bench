#!/usr/bin/env python3
"""ASR 基准：又小又快又准 —— 跑一批候选，算指标，画帕累托前沿。

用法::

    python3 scripts/bench.py --list                      # 有哪些候选
    python3 scripts/bench.py --engines vosk,fw-small     # 只跑指定几个
    python3 scripts/bench.py --all                       # 全跑（很慢，建议在 Codespace）
    python3 scripts/bench.py --all --repeats 3           # 每个测 3 次取中位数

设计原则
--------
* **每个候选在独立子进程里跑** —— 不同框架会互相污染（onnxruntime 与 ctranslate2
  抢线程、显存/内存泄漏），子进程隔离是唯一可靠的办法。
* **指标必须可复现**：冻结的 wav + 人工校对过的参考文本，CER 用标准编辑距离。
* 缺依赖/模型就**跳过并记明原因**，不让一个候选拖垮整轮。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio" / "course_zh_60s.wav"
REF = ROOT / "audio" / "reference_zh.txt"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """归一化：只留中文与字母数字。

    中文 ASR 的标点/空格各家不一（vosk 每个词之间都插空格），
    直接比字符串毫无意义，所以先剥掉所有非内容字符。
    另外统一成**简体**：whisper 在无引导时会输出繁体，
    不归一化会把「繁体 vs 简体」误算成识别错误。
    """
    keep = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            keep.append(ch)
    return "".join(keep)


try:  # 繁→简（可选依赖）
    from opencc import OpenCC

    _t2s = OpenCC("t2s").convert
except Exception:  # noqa: BLE001
    _t2s = None


def to_simplified(text: str) -> str:
    return _t2s(text) if _t2s else text


def cer(ref: str, hyp: str) -> float:
    """字符错误率（编辑距离 / 参考长度）。越小越准。"""
    ref, hyp = normalize(to_simplified(ref)), normalize(to_simplified(hyp))
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


# --------------------------------------------------------------------------
# 候选定义
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """一个可测的方案。"""

    key: str
    backend: str
    label: str
    size_mb: float = 0.0  # 运行期占用（包 + 模型），跑前实测填入
    cer: float = -1.0
    rtf: float = -1.0  # 实时倍率，越大越快
    load_s: float = 0.0
    status: str = "pending"
    note: str = ""
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


#: 包本体大小的粗略基准（实测填入；模型大小按实际文件算）
PACKAGE_MB = {
    "faster-whisper": 255.0,  # 精简后（去 av / onnxruntime）
    "vosk": 34.0,
    "sherpa-onnx": 40.0,
    "whisper.cpp": 6.0,  # 静态二进制
}

WHISPER_SIZES = {"tiny": 75, "base": 145, "small": 484, "medium": 1530}


def build_candidates(which: list[str] | None) -> list[Candidate]:
    out: list[Candidate] = []
    for size, mb in WHISPER_SIZES.items():
        for ct in ("int8", "float32"):
            key = f"fw-{size}-{ct}"
            out.append(
                Candidate(
                    key=key,
                    backend="faster-whisper",
                    label=f"faster-whisper {size}/{ct}",
                    size_mb=PACKAGE_MB["faster-whisper"] + mb,
                )
            )
    for size in ("small", "big"):
        out.append(
            Candidate(
                key=f"vosk-{size}",
                backend="vosk",
                label=f"vosk {'small-cn' if size=='small' else 'big-cn'}",
                size_mb=PACKAGE_MB["vosk"] + (66 if size == "small" else 1800),
            )
        )
    out.append(
        Candidate(
            key="sherpa-paraformer",
            backend="sherpa-onnx",
            label="sherpa-onnx paraformer-zh",
            size_mb=PACKAGE_MB["sherpa-onnx"] + 230,
        )
    )
    for size in ("tiny", "base", "small"):
        out.append(
            Candidate(
                key=f"wcpp-{size}",
                backend="whisper.cpp",
                label=f"whisper.cpp {size} (q5_0)",
                size_mb=PACKAGE_MB["whisper.cpp"] + {"tiny": 32, "base": 60, "small": 190}[size],
            )
        )
    if which:
        keys = {w.strip() for w in which}
        out = [c for c in out if c.key in keys or c.backend in keys]
    return out


# --------------------------------------------------------------------------
# 各后端的 runner（在子进程里执行）
# --------------------------------------------------------------------------


def run_candidate(c: Candidate, repeats: int, log=print) -> Candidate:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", c.key,
           "--repeats", str(repeats)]
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(MODELS / "hf"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env["PYTHONUNBUFFERED"] = "1"
    log(f"  → {c.label} …")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        c.status, c.note = "timeout", "超过 1 小时"
        return c
    if proc.returncode != 0:
        c.status = "failed"
        tail = (proc.stderr or proc.stdout or "")[-400:].strip()
        c.note = tail.splitlines()[-1] if tail else f"rc={proc.returncode}"
        return c
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        c.status, c.note = "failed", "子进程没输出 JSON"
        return c
    c.status = payload.get("status", "failed")
    c.note = payload.get("note", "")
    c.cer = payload.get("cer", -1.0)
    c.rtf = payload.get("rtf", -1.0)
    c.load_s = payload.get("load_s", 0.0)
    c.text = payload.get("text", "")
    if payload.get("size_mb"):
        c.size_mb = payload["size_mb"]
    return c


def _measure_mb(*paths: Path | str) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_file():
            total += p.stat().st_size
        else:
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / 1024 / 1024


# --------------------------------------------------------------------------
# 帕累托前沿
# --------------------------------------------------------------------------


def pareto_front(cands: list[Candidate]) -> list[Candidate]:
    """三维帕累托前沿：没有别的方案能同时在 CER（小）、size（小）、RTF（大）上都不更差。

    一个方案被支配的条件：存在另一个方案，``cer <= 它`` 且 ``size <= 它`` 且 ``rtf >= 它``，
    且至少一项严格更优。
    """
    ok = [c for c in cands if c.ok]
    front = []
    for a in ok:
        dominated = False
        for b in ok:
            if b is a:
                continue
            better_or_equal = (b.cer <= a.cer and b.size_mb <= a.size_mb and b.rtf >= a.rtf)
            strictly_better = (b.cer < a.cer or b.size_mb < a.size_mb or b.rtf > a.rtf)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda c: c.cer)


def write_outputs(cands: list[Candidate], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    front = pareto_front(cands)
    (outdir / "results.json").write_text(
        json.dumps({"candidates": [asdict(c) for c in cands],
                    "pareto": [c.key for c in front]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for c in sorted(cands, key=lambda x: (not x.ok, x.cer if x.ok else 9e9)):
        if c.ok:
            mark = "★" if c in front else " "
            rows.append(f"| {mark} | {c.label} | {c.cer*100:.2f}% | {c.size_mb:.0f} | {c.rtf:.2f}x | {c.load_s:.1f}s |")
        else:
            rows.append(f"| | {c.label} | — | {c.size_mb:.0f} | — | — | {c.status}: {c.note[:40]} |")

    md = [
        "# ASR 基准结果（准 > 小 > 快）",
        "",
        f"- 音频：`{AUDIO.name}`（60 秒中文课程讲课）",
        f"- 参考文本：{len(normalize(REF.read_text(encoding='utf-8')))} 字（人工校对）",
        "- ★ = 三维帕累托前沿上的方案",
        "- CER = 字符错误率（越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）",
        "",
        "| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## 帕累托前沿",
        "",
    ]
    if front:
        for c in front:
            md.append(f"- **{c.label}**：CER {c.cer*100:.2f}%，{c.size_mb:.0f}MB，{c.rtf:.2f}x")
    else:
        md.append("（本轮没有成功的方案）")
    (outdir / "pareto.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    _draw_svg(cands, front, outdir / "pareto.svg")


def _draw_svg(cands: list[Candidate], front: list[Candidate], path: Path) -> None:
    """手画一个 CER–体积 散点图（横轴体积，纵轴 CER，点大小/颜色表示速度）。

    不引 matplotlib：这个仓库要保持零依赖，而且手写 SVG 也就几十行。
    """
    ok = [c for c in cands if c.ok]
    if not ok:
        return
    W, H = 900, 560
    pad = 70
    max_mb = max(c.size_mb for c in ok) * 1.1
    max_cer = max(c.cer for c in ok) * 1.15 or 0.1
    max_rtf = max(c.rtf for c in ok) or 1.0

    def x(mb: float) -> float:
        return pad + (mb / max_mb) * (W - 2 * pad)

    def y(cer_: float) -> float:
        return H - pad - (cer_ / max_cer) * (H - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#fdfdfd"/>',
        f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="600">'
        f"ASR 帕累托：越靠左下越好（点越大越快）</text>",
        f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="13">'
        f"运行期占用 MB（包 + 模型）→</text>",
        f'<text x="16" y="{H/2}" font-size="13" transform="rotate(-90 16 {H/2})" '
        f'text-anchor="middle">CER 字符错误率 ↓（越低越准）</text>',
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#999"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#999"/>',
    ]
    front_keys = {c.key for c in front}
    for c in ok:
        r = 5 + 14 * (c.rtf / max_rtf)
        cx, cy = x(c.size_mb), y(c.cer)
        color = "#d62728" if c.key in front_keys else "#7f9fd0"
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}" '
            f'fill-opacity="0.55" stroke="{color}"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy - r - 5:.1f}" font-size="11" text-anchor="middle" '
            f'fill="#333">{c.label}</text>'
        )
    # 刻度
    for i in range(6):
        mb = max_mb * i / 5
        parts.append(f'<text x="{x(mb):.0f}" y="{H-pad+18}" font-size="10" '
                     f'text-anchor="middle" fill="#666">{mb:.0f}</text>')
        cer_ = max_cer * i / 5
        parts.append(f'<text x="{pad-8}" y="{y(cer_)+3:.0f}" font-size="10" '
                     f'text-anchor="end" fill="#666">{cer_*100:.0f}%</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------
# worker：真正跑某个候选（子进程入口）
# --------------------------------------------------------------------------


def worker(key: str, repeats: int) -> dict:
    """跑单个候选，把结果打进 stdout 的 JSON（最后一行）。"""
    import numpy as np

    def decode(path: Path, sr: int = 16000):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
             "-f", "s16le", "-"], capture_output=True, timeout=600,
        )
        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    audio = decode(AUDIO)
    seconds = len(audio) / 16000
    ref = REF.read_text(encoding="utf-8")

    def finish(status: str, note: str = "", text: str = "", rtf: float = -1.0,
               load_s: float = 0.0, size_mb: float = 0.0) -> dict:
        return {"status": status, "note": note, "text": text, "rtf": rtf,
                "load_s": load_s, "size_mb": size_mb,
                "cer": cer(ref, text) if text else -1.0}

    if key.startswith("fw-"):
        _, size, ct = key.split("-", 2)
        sys.path.insert(0, "/home/kibi/Work/UOOCPass")
        try:
            from uooc.asr import ensure_asr_importable

            ensure_asr_importable()
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            return finish("missing", f"faster-whisper 不可用：{exc}")
        try:
            t0 = time.time()
            model = WhisperModel(size, device="cpu", compute_type=ct)
            load_s = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            return finish("missing", f"模型 {size}/{ct} 加载失败：{str(exc)[:80]}")
        times, text = [], ""
        for _ in range(max(1, repeats)):
            t0 = time.time()
            segs, _ = model.transcribe(
                audio, language="zh", beam_size=5,
                initial_prompt="以下是普通话课程的简体中文讲稿，请用简体中文转写。",
            )
            text = "".join(s.text for s in segs)
            times.append(time.time() - t0)
        times.sort()
        rtf = seconds / times[len(times) // 2]
        return finish("ok", "", text, rtf, load_s, PACKAGE_MB["faster-whisper"] + WHISPER_SIZES[size])

    if key.startswith("vosk-"):
        size = key.split("-", 1)[1]
        model_dir = MODELS / ("vosk-model-small-cn-0.22" if size == "small" else "vosk-model-cn-0.22")
        if not model_dir.exists():
            return finish("missing", f"缺模型 {model_dir.name}（见 scripts/fetch_models.sh）")
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            return finish("missing", f"vosk 未安装：{exc}")
        SetLogLevel(-1)
        t0 = time.time()
        model = Model(str(model_dir))
        load_s = time.time() - t0
        pcm = (audio * 32767).astype("int16").tobytes()
        times, text = [], ""
        for _ in range(max(1, repeats)):
            t0 = time.time()
            rec = KaldiRecognizer(model, 16000)
            parts = []
            step = 8000
            for i in range(0, len(pcm), step):
                if rec.AcceptWaveform(pcm[i:i + step]):
                    parts.append(json.loads(rec.Result()).get("text", ""))
            parts.append(json.loads(rec.FinalResult()).get("text", ""))
            text = "".join(parts)
            times.append(time.time() - t0)
        times.sort()
        return finish("ok", "", text, seconds / times[len(times) // 2], load_s,
                      PACKAGE_MB["vosk"] + _measure_mb(model_dir))

    if key.startswith("wcpp-"):
        size = key.split("-", 1)[1]
        cli = os.environ.get("WHISPER_CLI") or "whisper-cli"
        model = MODELS / f"ggml-{size}-q5_0.bin"
        if not model.exists():
            return finish("missing", f"缺 ggml 模型 {model.name}")
        from shutil import which

        if not (Path(cli).is_file() or which(cli)):
            return finish("missing", f"找不到 whisper-cli（设 WHISPER_CLI 环境变量）")
        # whisper.cpp 只吃 16k wav 文件，先写一份临时 wav
        import tempfile
        import wave

        times, text = [], ""
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio * 32767).astype("int16").tobytes())
            for _ in range(max(1, repeats)):
                t0 = time.time()
                proc = subprocess.run(
                    [cli, "-m", str(model), "-f", str(wav), "-l", "zh",
                     "-otxt", "-of", str(Path(tmp) / "out"), "-np"],
                    capture_output=True, text=True, timeout=1800,
                )
                times.append(time.time() - t0)
                out_txt = Path(tmp) / "out.txt"
                if out_txt.exists():
                    text = out_txt.read_text(encoding="utf-8", errors="replace")
                else:
                    text = proc.stdout
        times.sort()
        return finish("ok", "", text, seconds / times[len(times) // 2], 0.0,
                      PACKAGE_MB["whisper.cpp"] + _measure_mb(model))

    if key.startswith("sherpa"):
        model_dir = MODELS / "sherpa-onnx-paraformer-zh-2023-09-14"
        if not model_dir.exists():
            return finish("missing", f"缺模型目录 {model_dir.name}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            return finish("missing", f"sherpa-onnx 未安装：{exc}")
        onnx = next(model_dir.glob("*.onnx"), None)
        tokens = next(model_dir.glob("tokens*.txt"), None)
        if not onnx or not tokens:
            return finish("missing", "模型目录里缺 .onnx / tokens.txt")
        t0 = time.time()
        rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(onnx), tokens=str(tokens),
            num_threads=2, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search",
        )
        load_s = time.time() - t0
        times, text = [], ""
        for _ in range(max(1, repeats)):
            t0 = time.time()
            stream = rec.create_stream()
            stream.accept_waveform(16000, audio)
            rec.decode_stream(stream)
            text = stream.result.text
            times.append(time.time() - t0)
        times.sort()
        return finish("ok", "", text, seconds / times[len(times) // 2], load_s,
                      PACKAGE_MB["sherpa-onnx"] + _measure_mb(model_dir))

    return finish("missing", f"未实现的候选：{key}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ASR 基准：又小又快又准")
    ap.add_argument("--worker", help=argparse.SUPPRESS)  # 子进程内部用
    ap.add_argument("--repeats", type=int, default=3, help="每个候选重复次数，取中位数")
    ap.add_argument("--engines", help="只跑这些（逗号分隔，可用 key 或 backend 名）")
    ap.add_argument("--all", action="store_true", help="跑全部候选")
    ap.add_argument("--list", action="store_true", help="只列出候选")
    ap.add_argument("--out", default=str(RESULTS), help="结果输出目录")
    args = ap.parse_args(argv)

    if args.worker:
        try:
            payload = worker(args.worker, args.repeats)
        except Exception as exc:  # noqa: BLE001
            payload = {"status": "failed", "note": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    which = args.engines.split(",") if args.engines else None
    if not args.all and not which and not args.list:
        ap.print_help()
        return 0

    cands = build_candidates(which)
    if args.list:
        print(f"{len(cands)} 个候选：")
        for c in cands:
            print(f"  {c.key:<22} {c.label:<34} ~{c.size_mb:.0f}MB")
        return 0

    print(f"跑 {len(cands)} 个候选（每个 {args.repeats} 次取中位数）\n")
    for c in cands:
        c = run_candidate(c, args.repeats)
        if c.ok:
            print(f"    ✓ CER {c.cer*100:5.2f}%  {c.size_mb:>5.0f}MB  {c.rtf:5.2f}x  加载 {c.load_s:.1f}s")
            print(f"      {normalize(c.text)[:70]}")
        else:
            print(f"    ✗ {c.status}: {c.note[:100]}")

    write_outputs(cands, Path(args.out))
    front = pareto_front(cands)
    print(f"\n帕累托前沿（{len(front)} 个）：")
    for c in front:
        print(f"  ★ {c.label:<34} CER {c.cer*100:5.2f}%  {c.size_mb:>5.0f}MB  {c.rtf:5.2f}x")
    print(f"\n结果写到 {args.out}/（results.json / pareto.md / pareto.svg）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
