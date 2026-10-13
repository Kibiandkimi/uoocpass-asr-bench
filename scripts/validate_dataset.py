#!/usr/bin/env python3
"""数据集校验：确认每段的**音频内容**和**参考文本**真的对得上。

为什么需要它
------------
建数据集时我只检查了「SRT 末条时间戳 ≈ 音频时长」，以为这样就能保证对齐。
**不够** —— 实测大学英语有节课：视频 257 秒、SRT 260 秒（只差 2.1 秒，检查通过），
但内容完全是别的东西（ASR 输出讲 education，字幕讲「学习方式」）。
时长吻合只是**必要条件**，不是充分条件。

做法
----
用最快的 ASR（whisper.cpp tiny 或 faster-whisper tiny）跑一遍音频，跟参考文本算
CER。真正对齐的片段 CER 会很低；错配的会非常高。给一个阈值判定。

    python3 scripts/validate_dataset.py                    # 用默认最快的模型
    python3 scripts/validate_dataset.py --engine sherpa-sensevoice
    python3 scripts/validate_dataset.py --fix               # 把不通过的标成 rejected

输出 ``audio/validation.json``，并把结论写回 ``audio/dataset.json`` 各段的
``content_check`` 字段（不删数据，只标记 —— 保留现场便于复查）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"

#: 判定阈值：真正对齐的片段 CER 通常 < 15%；错配的基本都 > 40%
PASS_CER = 0.25
SUSPECT_CER = 0.45


def transcribe_tiny(wav: Path, model_size: str = "tiny") -> str:
    """用 faster-whisper 最快档转写（语言自动）。"""
    bench.install_av_stub()
    from faster_whisper import WhisperModel
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True, timeout=900,
    )
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _info = model.transcribe(audio, beam_size=1, vad_filter=False)
    return "".join(s.text for s in segs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="校验数据集内容对齐")
    ap.add_argument("--dataset", default="audio/dataset.json")
    ap.add_argument("--model", default="tiny", help="校验用的 faster-whisper 档位")
    ap.add_argument("--fix", action="store_true", help="把不通过的段标成 rejected")
    args = ap.parse_args(argv)

    bench.DATASET = ROOT / args.dataset
    clips = bench.load_clips()
    if not clips:
        print(f"{args.dataset} 里没有可用片段", file=sys.stderr)
        return 1

    print(f"校验 {len(clips)} 段（用 faster-whisper {args.model}）\n")
    report = {"pass": [], "suspect": [], "reject": [], "rows": []}
    for clip in clips:
        try:
            hyp = transcribe_tiny(clip.wav, args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"  ? {clip.id:<22} 转写失败：{type(exc).__name__}: {exc}")
            report["suspect"].append(clip.id)
            continue
        # 参考文本可能含时间轴残留，统一走同一套归一化
        score = bench.cer(clip.reference, hyp)
        if score <= PASS_CER:
            verdict = "pass"
        elif score <= SUSPECT_CER:
            verdict = "suspect"
        else:
            verdict = "reject"
        report[verdict].append(clip.id)
        report["rows"].append({"id": clip.id, "cer": score, "verdict": verdict,
                               "hyp_head": bench.normalize(hyp)[:80]})
        mark = {"pass": "✓", "suspect": "?", "reject": "✗"}[verdict]
        print(f"  {mark} {clip.id:<22} 内容 CER {score*100:5.1f}%  {clip.domain}")

    print(f"\n通过 {len(report['pass'])}，可疑 {len(report['suspect'])}，"
          f"不合格 {len(report['reject'])}")
    if report["reject"]:
        print("\n不合格的段（音频与参考不是同一内容，必须剔除）：")
        for cid in report["reject"]:
            print(f"  ✗ {cid}")

    (AUDIO / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fix:
        data = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
        for c in data["clips"]:
            row = next((r for r in report["rows"] if r["id"] == c["id"]), None)
            if not row:
                continue
            c["content_check"] = {"cer": round(row["cer"], 4), "verdict": row["verdict"]}
            if row["verdict"] == "reject":
                c["verified"] = "rejected-mismatch"
                c["note"] = ("内容校验不通过：音频与参考文本不是同一内容"
                             "（时长虽接近但内容错配）。不可用于 CER 统计。")
        kept = [c for c in data["clips"] if c["verified"] != "rejected-mismatch"]
        secs = sum(c["seconds"] for c in kept)
        data["note"] += f"\n\n内容校验（scripts/validate_dataset.py）：剔除错配段后剩 {len(kept)} 段 / {secs} 秒。"
        (ROOT / args.dataset).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写回 {args.dataset}：有效 {len(kept)} 段 / {secs} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
