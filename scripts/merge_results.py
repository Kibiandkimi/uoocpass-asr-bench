#!/usr/bin/env python3
"""把并排跑出来的各 job 结果合并，重算全局帕累托前沿。

为什么需要它：GitHub Actions 里每个候选是一个独立 job（天然隔离、并排跑），
各自只看到自己那份 ``results.json``。哪个方案真正在**全局**帕累托前沿上，
必须等所有 job 回来才知道 —— 就是这个脚本干的活。

用法::

    python3 scripts/merge_results.py --root results --out results/final
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import Candidate, pareto_front, write_outputs  # noqa: E402


def load(root: Path) -> list[Candidate]:
    """从 ``root`` 下所有 ``results.json``/``*.json`` 里收集候选。

    同一个 key 出现多次就取**最好的一次**（CER 最小），
    因为并行 job 可能因为机器差异跑出不同速度。
    """
    best: dict[str, Candidate] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for raw in payload.get("candidates") or []:
            try:
                c = Candidate(**raw)
            except TypeError:
                continue
            if not c.ok:
                continue
            old = best.get(c.key)
            if old is None or c.cer < old.cer:
                best[c.key] = c
    return list(best.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合并并行基准结果")
    ap.add_argument("--root", default="results", help="各 job 结果目录")
    ap.add_argument("--out", default="results/final", help="合并输出目录")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    cands = load(root)
    if not cands:
        print(f"在 {root} 下没找到任何成功结果", file=sys.stderr)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pareto.md").write_text("# ASR 基准结果\n\n（本轮没有成功的方案）\n", encoding="utf-8")
        return 1

    print(f"合并了 {len(cands)} 个候选：")
    for c in sorted(cands, key=lambda x: x.cer):
        warn = "" if c.weakest_verified in ("human", "subtitle") else f"  ⚠ 参考可信度={c.weakest_verified}"
        print(f"  {c.label:<38} CER {c.cer*100:5.2f}%  {c.size_mb:>6.0f}MB  "
              f"{c.rtf:5.2f}x  {c.clips}段/{c.clip_seconds:.0f}s{warn}")

    write_outputs(cands, out)
    front = pareto_front(cands)
    print(f"\n全局帕累托前沿（{len(front)} 个）：")
    for c in front:
        print(f"  ★ {c.label:<36} CER {c.cer*100:5.2f}%  {c.size_mb:>6.0f}MB  {c.rtf:5.2f}x")

    # 给个「按优先级选」的建议 —— 准 > 小 > 快
    print("\n按「准 > 小 > 快」的推荐：")
    if front:
        acc = min(front, key=lambda c: c.cer)
        small = min(front, key=lambda c: c.size_mb)
        fast = max(front, key=lambda c: c.rtf)
        print(f"  最准    : {acc.label}（CER {acc.cer*100:.2f}%）")
        print(f"  最小    : {small.label}（{small.size_mb:.0f}MB，CER {small.cer*100:.2f}%）")
        print(f"  最快    : {fast.label}（{fast.rtf:.2f}x，CER {fast.cer*100:.2f}%）")
        # 均衡：三项都归一化后取最小值之和
        def balance(c: Candidate) -> float:
            cer_n = c.cer / max(x.cer for x in front)
            size_n = c.size_mb / max(x.size_mb for x in front)
            rtf_n = max(x.rtf for x in front) / c.rtf
            return cer_n + size_n + rtf_n

        best = min(front, key=balance)
        print(f"  最均衡  : {best.label}（CER {best.cer*100:.2f}% / "
              f"{best.size_mb:.0f}MB / {best.rtf:.2f}x）")
    print(f"\n写到 {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
