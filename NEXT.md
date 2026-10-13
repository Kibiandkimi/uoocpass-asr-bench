# 下一步（按优先级）

## P0：数据集对齐（阻断项）

现在 **4 段 328 秒，但只有 148 秒可信**。两段标了 `subtitle-misaligned`。

**根因**：`/home/learn/getUnitLearn` 给的 `video_url`（mp4）与 `subtitle`（srt）
**不是同一条时间轴** —— 视频带片头，SRT 从讲稿起点计时。实测「穷举法」那节：
字幕 62s 处讲「国王的婚姻 / 旅行商问题」，而 mp4 60s 处音频在讲「3x+y+z=100」。
扫 ±180s 固定偏移也救不回来（CER 仍 94%），所以是**非恒定偏移**。

**修法：按内容对齐，不按时间戳切。**
1. 用最好的模型（SenseVoice）转写整段视频 → `hyp`；
2. 把完整 SRT 文本当 `ref`；
3. 用**滑动窗口 + 编辑距离**在 `ref` 上找与 `hyp` 最匹配的窗口；
4. 以该窗口对应的时间范围反过来切音频（或直接用整段，让 CER 在整段上算）。

更省事的替代：**不做切片，整段视频配整段字幕**，CER 在完整视频上算（micro），
这样根本没有对齐问题。代价是每段音频长（几分钟 → CI 时间变长）。

## P1：把真正强的模型加进来（当前候选是「矮子里拔将军」）

已确认漏掉整个中文专用版图。按公开基准（184 段真实普通话 + 人工参考）：

| 模型 | 中文 CER | 速度 | 体积 | 备注 |
| --- | --- | --- | --- | --- |
| **SenseVoice-Small** | **7.81%** | ~20x (CPU 17.2x) | 449MB (f16) | 编码器+CTC，一次前向 |
| Fun-ASR-Nano | 8.06–8.30% | 340x (vLLM) | enc + Qwen3-0.6B GGUF | LLM 解码，最准 |
| Paraformer | 9.85–10.18% | ~21x | 401MB | 已在测 |
| whisper.cpp base | **31.33%** | 9.9x | 142MB | |
| whisper.cpp small | **22.12%** | 4.6x | 466MB | |

**中文上 FunASR 比 whisper.cpp 准 2.7 倍**（其自带基准原话）。
`sherpa-onnx` 已支持：`sense_voice` / `fire_red_asr_ctc` / `funasr_nano` /
`zipformer` / `qwen3_asr` / `paraformer`。此外还有 FireRedASR（小红书，工业级）、
Qwen3-ASR-0.6B。

要加的候选（按预期收益排序）：
1. `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`（**必加**，预期 CER 最低）
2. `sherpa-onnx-paraformer-zh-2024-03-09`（比现在用的 2023-09-14 新）
3. `sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16`
4. `sherpa-onnx-funasr-nano-*`（若已发布转换版）
5. `sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03`
6. Qwen3-ASR-0.6B int4（HF 上有人转）

## P2：CI 模型下载失败（未解决）

`fetch_models.sh` 在 GitHub runner 上失败，但**同一个 runner 直接 curl 成功**
（`gh run view 38411380801` 的日志证明：190MB 和 43MB 都下到了）。
脚本立即失败且无输出，`bash -x` 的跟踪也没落到日志里。排查方向见 `NOTES.md`。

**换个更稳的路子**：不要用 bash + curl 拼，直接让 Python 下 —— 用
`huggingface_hub.snapshot_download` 或 `urllib` + 重试，错误信息天然可读。

## P3：数据集规模

目标：**≥ 30 分钟、≥ 10 段、跨 ≥ 4 个领域**（理工 / 人文 / 英文 / 带口音或快语速）。
- 大学计算机 192/230 节有字幕（最容易做大的来源）
- 大学英语 13/51 节有字幕
- 艺术美学 0/57 节 → 只能靠 ASR 转，且**不能当参考**（循环论证）
