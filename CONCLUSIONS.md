# 结论（终版，口径已修正）

数据集：**9 段 / 1666 秒（27.8 分钟）**，全部通过内容校验，参考 = 平台官方字幕。
候选：11 个（Whisper 系 / FunASR 系 / vosk 系），全部实测。

## 排名（micro-CER / macro-CER / 体积 / 速度）

| 候选 | micro | macro | 体积 | RTF |
| --- | --- | --- | --- | --- |
| **whisper.cpp base (q5_0)** ★ | **3.71%** | **3.69%** | **63MB** | 5.75x |
| whisper.cpp small (q5_0) | 3.82% | 4.12% | 187MB | 4.00x |
| faster-whisper base/int8 | 3.87% | 4.12% | 400MB | 10.40x |
| faster-whisper small/int8 | 4.44% | 7.06% | 739MB | 4.06x |
| whisper.cpp tiny (q5_0) | 7.29% | 9.82% | 37MB | 8.91x |
| faster-whisper tiny/int8 | 8.52% | 9.17% | 330MB | 15.40x |
| sherpa SenseVoice-Small | 13.74% | 8.67% | 1164MB | 9.46x |
| sherpa paraformer-zh (2023-09) | 25.28% | 18.95% | 274MB | 14.27x |
| sherpa paraformer-zh (2024-03) | 29.91% | 21.42% | 1043MB | 14.65x |
| vosk big-cn | 63.41% | 38.23% | 2078MB | 2.33x |
| vosk small-cn | 65.75% | 42.18% | 99MB | 4.64x |

## 帕累托前沿（micro-CER / 体积 / 速度）

| 方案 | CER | 体积 | 速度 | 定位 |
| --- | --- | --- | --- | --- |
| **whisper.cpp base** | 3.71% | **63MB** | 5.75x | **最均衡，推荐** |
| faster-whisper base | 3.87% | 400MB | **10.40x** | 要更快 |
| whisper.cpp tiny | 7.29% | **37MB** | 8.91x | 极致省空间 |
| faster-whisper tiny | 8.52% | 330MB | 15.40x | 最快 |
| paraformer-zh (2023-09) | 25.28% | 274MB | 14.27x | 不推荐 |

**推荐 `whisper.cpp base (q5_0)`**：63MB、CER 3.71%（与 739MB 的 fw-small 只差 0.7 个点），
比当前 UOOCPass 在用的 faster-whisper small 省 **676MB**。

## 数据集的局限（必须写明，否则数字会被误用）

1. **只有两类领域**：理工科讲课 + 英文讲课。**缺**人文艺术、快语速、方言口音、
   背景音乐 —— 这些才是拉开模型差距的场景。
2. **朗读式讲课为主**，语音难度低于口语/会议/电话。所以「各模型差距小」不代表
   在难场景下也小。与 FunASR 官方基准（口语/会议，SenseVoice 7.81% vs
   whisper.cpp base 31.33%）的差异，**主因极可能是领域不同**，不是谁测错了。
3. **是平台官方字幕**，不是独立人工转写。它与平台自身可能的转写流程相关度高。

## 待查

- **SenseVoice 在 en-2（中英混说）上 60%**：单语模型处理 code-switching 的
  真实局限，还是 `language="auto"` 中途漂移？需要拿 en-2 单独扫
  language=auto/en/zh 三档定位。
- **paraformer 中文段也有 20~43%**，与它「中文专精」的定位不符 —— 可能同样是
  用法问题（`decoding_method` / 分段策略），而不是模型能力。这两点排除前，
  不对 FunASR 系下最终判决。
