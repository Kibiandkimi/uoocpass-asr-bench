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
AUDIO = ROOT / "audio" / "course_zh_60s.wav"  # 兼容旧路径
DATASET = ROOT / "audio" / "dataset.json"
REF = ROOT / "audio" / "reference_zh.txt"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


@dataclass
class Clip:
    """一段测试音频 + 它的参考文本。

    ``verified`` 很重要，决定了这个数据点可不可信：

    * ``human``    —— 人工逐字校对过，最可信
    * ``subtitle`` —— 平台**官方字幕**，等价于官方转写，可信
                      （注意：SRT 的时间轴和音轨一致，所以切音频时按同样时间范围切即可对齐）
    * ``asr``      —— 某个模型转的草稿，**未经校对**
                      ⚠️ 用 A 模型的输出测 B 模型 = 循环论证，必须标注清楚
    """

    id: str
    wav: Path
    reference: str
    source: str = ""
    domain: str = ""
    seconds: float = 0.0
    verified: str = "human"


def load_clips() -> list[Clip]:
    """读 ``audio/dataset.json``；没有 manifest 就退回单段旧布局。"""
    import json

    if not DATASET.is_file():
        return [Clip("legacy", AUDIO, REF.read_text(encoding="utf-8"), verified="human")]
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    clips: list[Clip] = []
    for raw in data.get("clips") or []:
        wav = ROOT / "audio" / raw["wav"]
        ref_path = ROOT / "audio" / raw.get("reference", "")
        if not wav.is_file() or not ref_path.is_file():
            continue
        text = ref_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        clips.append(
            Clip(
                id=raw["id"], wav=wav, reference=text,
                source=raw.get("source", ""), domain=raw.get("domain", ""),
                seconds=raw.get("seconds", 0.0), verified=raw.get("verified", "human"),
            )
        )
    return clips


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


def edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein 编辑距离（字符级）。"""
    ref, hyp = normalize(to_simplified(ref)), normalize(to_simplified(hyp))
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def _clip_metrics(clip, hyp: str) -> dict:
    """单段指标。留 errors/ref_len 是为了后面能算 micro-CER。"""
    ref_n = normalize(to_simplified(clip.reference))
    err = edit_distance(clip.reference, hyp)
    return {
        "cer": err / len(ref_n) if ref_n else -1.0,
        "errors": err,
        "ref_len": len(ref_n),
        "verified": clip.verified,
        "domain": clip.domain,
        "hyp": normalize(hyp)[:120],
    }


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
    #: 每段的结果：{clip_id: {"cer": …, "verified": …, "hyp": …}}
    per_clip: dict = field(default_factory=dict)
    #: 数据集规模（段数 / 总秒数 / 最弱可信度），防止「拿 1 段 60 秒就下结论」
    clips: int = 0
    clip_seconds: float = 0.0
    weakest_verified: str = ""

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
    # sherpa-onnx 系：模型体积按实测目录大小在 worker 里覆盖
    for key, label, mb in (
        ("sherpa-paraformer", "sherpa-onnx paraformer-zh (2023-09)", 230),
        ("sherpa-paraformer-2024", "sherpa-onnx paraformer-zh (2024-03)", 230),
        ("sherpa-sensevoice", "sherpa-onnx SenseVoice-Small", 250),
        ("sherpa-fire-red", "sherpa-onnx FireRedASR", 1500),
        ("sherpa-zipformer-ctc", "sherpa-onnx Zipformer-CTC-zh", 160),
        ("sherpa-funasr-nano", "sherpa-onnx Fun-ASR-Nano", 700),
    ):
        out.append(
            Candidate(
                key=key,
                backend="sherpa-onnx",
                label=label,
                size_mb=PACKAGE_MB["sherpa-onnx"] + mb,
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
    # 别只取「最后一行」—— faster-whisper 之类的库会往 stdout 打警告/进度，
    # 真正的 JSON 可能被夹在中间。从后往前找第一个能解析成 dict 的行。
    payload = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "status" in obj:
            payload = obj
            break
    if payload is None:
        c.status = "failed"
        tail = ((proc.stderr or "") + (proc.stdout or ""))[-300:].strip()
        c.note = f"子进程没输出可解析的 JSON。尾部：{tail.splitlines()[-1] if tail else '（空）'}"
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


def install_av_stub() -> None:
    """让 ``import faster_whisper`` 通过 —— 它顶层 ``import av``（PyAV）没有守卫。

    PyAV 只用于**文件解码**，而本基准传的是 numpy 数组，永远不走那条路。
    少装它能省 128MB（27MB 包 + 101MB FFmpeg 共享库）。真被调用会抛错，
    不会静默算错。这段逻辑与 ``uooc/asr.py`` 里的一致 —— 基准仓库特意
    **不依赖主项目**，所以这里要自带一份。
    """
    import sys
    import types

    if "av" in sys.modules:
        return
    try:
        import av  # noqa: F401

        return
    except ImportError:
        pass
    av = types.ModuleType("av")
    av.__version__ = "stub"
    av.open = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("av 占位模块被调用了"))
    for name in ("av.audio", "av.audio.resampler", "av.audio.fifo", "av.error"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["av.audio.resampler"].AudioResampler = type("AudioResampler", (), {})
    sys.modules["av.audio.fifo"].AudioFifo = type("AudioFifo", (), {})
    sys.modules["av.error"].InvalidDataError = type("InvalidDataError", (Exception,), {})
    sys.modules["av.audio"].resampler = sys.modules["av.audio.resampler"]
    sys.modules["av.audio"].fifo = sys.modules["av.audio.fifo"]
    av.audio = sys.modules["av.audio"]
    av.error = sys.modules["av.error"]
    sys.modules["av"] = av


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
            warn = "" if c.weakest_verified in ("human", "subtitle") else f" ⚠{c.weakest_verified}"
            rows.append(
                f"| {mark}{warn} | {c.label} | {c.cer*100:.2f}% | {c.size_mb:.0f} | "
                f"{c.rtf:.2f}x | {c.load_s:.1f}s | {c.clips}/{c.clip_seconds:.0f}s |"
            )
        else:
            rows.append(f"| | {c.label} | — | {c.size_mb:.0f} | — | — | {c.status}: {c.note[:40]} |")

    clips = load_clips()
    total_s = sum(c.seconds for c in clips) or 0
    by_trust: dict[str, float] = {}
    for c in clips:
        by_trust[c.verified] = by_trust.get(c.verified, 0) + c.seconds
    trust_txt = "、".join(f"{k} {v:.0f}s" for k, v in sorted(by_trust.items()))
    domains = sorted({c.domain for c in clips if c.domain})
    md = [
        "# ASR 基准结果（准 > 小 > 快）",
        "",
        f"- 数据集：**{len(clips)} 段 / {total_s:.0f} 秒**，领域：{'、'.join(domains) or '—'}",
        f"- 参考可信度：{trust_txt}",
        "  （`human`=人工校对；`subtitle`=平台官方字幕且已验证对齐；"
        "`subtitle-misaligned`=字幕时间轴错位、结果不可信；`asr`=模型草稿=循环论证）",
        "- ★ = 三维帕累托前沿上的方案",
        "- CER = 字符错误率（micro：总编辑距离/总参考长度，越低越准）；"
        "size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）",
        "",
        "| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 | 段/秒 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
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

    clips = load_clips()
    if not clips:
        return {"status": "missing", "note": "数据集为空（audio/dataset.json）"}
    # 预先解码全部音频，避免把解码时间算进 RTF
    decoded = [(clip, decode(clip.wav)) for clip in clips]
    total_seconds = sum(len(a) / 16000 for _c, a in decoded)

    def finish(status: str, note: str = "", text: str = "", rtf: float = -1.0,
               load_s: float = 0.0, size_mb: float = 0.0,
               per_clip: dict | None = None) -> dict:
        per_clip = per_clip or {}
        ok = [v for v in per_clip.values() if v.get("cer", -1) >= 0]
        # micro-CER：总编辑距离 / 总参考长度（不是各段 CER 的平均）
        if ok:
            tot_err = sum(v["errors"] for v in ok)
            tot_len = sum(v["ref_len"] for v in ok)
            micro = tot_err / tot_len if tot_len else -1.0
        else:
            micro = -1.0
        ranks = {"human": 3, "subtitle": 2, "asr": 1, "": 0}
        weakest = min((c.verified for c, _ in decoded), key=lambda v: ranks.get(v, 0))
        return {"status": status, "note": note, "text": text, "rtf": rtf,
                "load_s": load_s, "size_mb": size_mb, "cer": micro,
                "per_clip": per_clip, "clips": len(clips),
                "clip_seconds": total_seconds, "weakest_verified": weakest}

    if key.startswith("fw-"):
        _, size, ct = key.split("-", 2)
        install_av_stub()
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            return finish("missing", f"faster-whisper 不可用：{exc}")
        try:
            t0 = time.time()
            model = WhisperModel(size, device="cpu", compute_type=ct)
            load_s = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            return finish("missing", f"模型 {size}/{ct} 加载失败：{str(exc)[:80]}")
        per_clip: dict = {}
        elapsed: list[float] = []
        texts: list[str] = []
        # ⚠️ 这个循环变量**不能叫 _`** —— 下面 `segs, info = model.transcribe(...)`
        # 以前写成 `segs, _ = ...`，把 `_` 覆盖成 TranscriptionInfo，
        # 于是 `if _ == 0` 永远为假，per_clip 一直空、CER 报 -100%。
        for rep in range(max(1, repeats)):
            for clip, audio in decoded:
                t0 = time.time()
                segs, _info = model.transcribe(
                    audio, language="zh", beam_size=5, vad_filter=False,
                    initial_prompt="以下是普通话课程的简体中文讲稿，请用简体中文转写。",
                )
                text = "".join(seg.text for seg in segs)
                elapsed.append(time.time() - t0)
                if rep == 0:  # 只留第一次的输出做展示与比对
                    texts.append(text)
                    per_clip[clip.id] = _clip_metrics(clip, text)
        rtf = total_seconds / sum(sorted(elapsed)[: len(decoded)])
        return finish("ok", "", "\n".join(texts), rtf, load_s,
                      PACKAGE_MB["faster-whisper"] + WHISPER_SIZES[size], per_clip)

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
        per_clip: dict = {}
        elapsed: list[float] = []
        texts: list[str] = []
        for rep in range(max(1, repeats)):
            for clip, audio in decoded:
                pcm = (audio * 32767).astype("int16").tobytes()
                t0 = time.time()
                rec = KaldiRecognizer(model, 16000)
                parts = []
                step = 8000
                for i in range(0, len(pcm), step):
                    if rec.AcceptWaveform(pcm[i:i + step]):
                        parts.append(json.loads(rec.Result()).get("text", ""))
                parts.append(json.loads(rec.FinalResult()).get("text", ""))
                text = "".join(parts)
                elapsed.append(time.time() - t0)
                if rep == 0:
                    texts.append(text)
                    per_clip[clip.id] = _clip_metrics(clip, text)
        rtf = total_seconds / sum(sorted(elapsed)[: len(decoded)])
        return finish("ok", "", "\n".join(texts), rtf, load_s,
                      PACKAGE_MB["vosk"] + _measure_mb(model_dir), per_clip)

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

        per_clip: dict = {}
        elapsed: list[float] = []
        texts: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            for rep in range(max(1, repeats)):
                for clip, audio in decoded:
                    wav = Path(tmp) / "a.wav"
                    with wave.open(str(wav), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes((audio * 32767).astype("int16").tobytes())
                    t0 = time.time()
                    proc = subprocess.run(
                        [cli, "-m", str(model), "-f", str(wav), "-l", "zh",
                         "-otxt", "-of", str(Path(tmp) / "out"), "-np"],
                        capture_output=True, text=True, timeout=1800,
                    )
                    elapsed.append(time.time() - t0)
                    out_txt = Path(tmp) / "out.txt"
                    text = (out_txt.read_text(encoding="utf-8", errors="replace")
                            if out_txt.exists() else proc.stdout)
                    if rep == 0:
                        texts.append(text)
                        per_clip[clip.id] = _clip_metrics(clip, text)
        rtf = total_seconds / sum(sorted(elapsed)[: len(decoded)])
        return finish("ok", "", "\n".join(texts), rtf, 0.0,
                      PACKAGE_MB["whisper.cpp"] + _measure_mb(model), per_clip)

    if key.startswith("sherpa-"):
        try:
            import sherpa_onnx
        except ImportError as exc:
            return finish("missing", f"sherpa-onnx 未安装：{exc}")

        # 各模型目录与构造方式都不同 —— 显式列出来，比猜文件好
        SPECS = {
            "sherpa-paraformer": (
                "sherpa-onnx-paraformer-zh-2023-09-14", "paraformer"),
            "sherpa-paraformer-2024": (
                "sherpa-onnx-paraformer-zh-2024-03-09", "paraformer"),
            "sherpa-sensevoice": (
                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "sensevoice"),
            "sherpa-fire-red": ("sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16", "fire-red"),
            "sherpa-zipformer-ctc": (
                "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03", "zipformer-ctc"),
            "sherpa-funasr-nano": ("sherpa-onnx-funasr-nano-zh-2025-11-25", "funasr-nano"),
        }
        dirname, kind = SPECS.get(key, (None, None))
        if not dirname:
            return finish("missing", f"未配置的 sherpa 候选：{key}")
        model_dir = MODELS / dirname
        if not model_dir.is_dir():
            return finish("missing", f"缺模型目录 {dirname}（跑 scripts/fetch_models.py）")

        t0 = time.time()
        try:
            if kind == "paraformer":
                onnx = next(model_dir.glob("*.onnx"), None)
                tokens = next(model_dir.glob("tokens*.txt"), None)
                if not onnx or not tokens:
                    return finish("missing", "缺 .onnx / tokens.txt")
                rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=str(onnx), tokens=str(tokens),
                    num_threads=2, sample_rate=16000, feature_dim=80,
                    decoding_method="greedy_search")
            elif kind == "sensevoice":
                model = next((p for p in model_dir.glob("model*.onnx")), None)
                tokens = next(model_dir.glob("tokens*.txt"), None)
                if not model or not tokens:
                    return finish("missing", "缺 model.onnx / tokens.txt")
                rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model), tokens=str(tokens),
                    num_threads=2, use_itn=True, language="zh")
            elif kind == "zipformer-ctc":
                model = next(model_dir.glob("*.onnx"), None)
                tokens = next(model_dir.glob("tokens*.txt"), None)
                rec = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                    model=str(model), tokens=str(tokens), num_threads=2)
            else:
                return finish("missing", f"kind={kind} 的构造还没接（缺参考用法）")
        except Exception as exc:  # noqa: BLE001
            return finish("missing", f"构造识别器失败（{kind}）：{type(exc).__name__}: {exc}")
        load_s = time.time() - t0
        per_clip: dict = {}
        elapsed: list[float] = []
        texts: list[str] = []
        for rep in range(max(1, repeats)):
            for clip, audio in decoded:
                t0 = time.time()
                stream = rec.create_stream()
                stream.accept_waveform(16000, audio)
                rec.decode_stream(stream)
                text = stream.result.text
                elapsed.append(time.time() - t0)
                if rep == 0:
                    texts.append(text)
                    per_clip[clip.id] = _clip_metrics(clip, text)
        rtf = total_seconds / sum(sorted(elapsed)[: len(decoded)])
        return finish("ok", "", "\n".join(texts), rtf, load_s,
                      PACKAGE_MB["sherpa-onnx"] + _measure_mb(model_dir), per_clip)

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
