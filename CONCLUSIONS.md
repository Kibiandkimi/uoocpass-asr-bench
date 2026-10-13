# 结论（终版）

数据集：**9 段 / 1666 秒（27.8 分钟）**，参考 = 平台官方字幕，全部通过内容校验。
其中 1 段是**中英混说**（en-1911877932-2），单独看，见下。

## 排名

### A. 8 段单语（3 段英文 + 5 段中文理工）—— 主榜

| 候选 | micro-CER | macro-CER | 体积 | RTF |
| --- | --- | --- | --- | --- |
| **sherpa SenseVoice-Small** | **1.85%** | **2.19%** | 1164MB | 9.46x |
| **whisper.cpp base (q5_0)** | 2.80% | 3.24% | **63MB** | 5.75x |
| whisper.cpp small (q5_0) | 3.03% | 3.77% | 187MB | 4.00x |
| faster-whisper base/int8 | 3.10% | 3.78% | 400MB | 10.40x |
| faster-whisper small/int8 | 3.89% | 7.11% | 739MB | 4.06x |
| faster-whisper tiny/int8 | 5.77% | 7.90% | 330MB | 15.40x |
| whisper.cpp tiny (q5_0) | 6.49% | 9.73% | **37MB** | 8.91x |
| sherpa paraformer-zh (2023-09) | 20.74% | 15.93% | 274MB | 14.27x |
| sherpa paraformer-zh (2024-03) | 24.64% | 17.77% | 1043MB | 14.65x |
| vosk big-cn | 54.30% | 30.61% | 2078MB | 2.33x |
| vosk small-cn | 57.22% | 35.05% | 99MB | 4.64x |

### B. 含中英混说段（9 段）—— 看 code-switching 影响

| 候选 | micro | macro |
| --- | --- | --- |
| whisper.cpp base | 3.71% | 3.69% |
| **sherpa SenseVoice-Small** | **13.74%** | **8.67%** |

SenseVoice 从 2.19% 掉到 8.67%，**全部来自那一句话**（记录片里的中国课堂夹中文数字
「九五四八四」）。Whisper 系因为原生多语言，几乎不受影响 —— 这是**真实的能力差异**，
不是 bug。

## 帕累托前沿（8 段单语）

| 方案 | CER | 体积 | 速度 | 定位 |
| --- | --- | --- | --- | --- |
| **SenseVoice-Small** | **1.85%** | 1164MB | 9.46x | 最准 |
| **whisper.cpp base** | 2.80% | **63MB** | 5.75x | **最均衡** |
| whisper.cpp tiny | 6.49% | **37MB** | 8.91x | 极致小 |
| faster-whisper base | 3.10% | 400MB | **10.40x** | 最快之一 |

**两个都值得选**，取决于约束：

* 空间不敏感、要最准 → **SenseVoice-Small**（1.85%，但 1164MB）
* 要省空间（本项目 `/home` 常年只剩几百 MB）→ **whisper.cpp base**（2.80%，63MB）
* code-switching 场景多 → **Whisper 系**（SenseVoice 会崩）

## 定向实验：SenseVoice 的 language 参数

同一批单语片段，只改 `language`：

| language | micro | macro |
| --- | --- | --- |
| auto | 1.83% | 2.19% |
| en | 1.88% | 2.21% |
| zh | 1.83% | 2.19% |

**单语片段上三档几乎无差别** —— 所以「语言写死导致冤枉」这个假设**不成立**
（真正的原因是混说段）。bench 里按片段给 language / 混合时取 auto 的做法仍然保留，
因为它对混说场景是对的。

## 我踩过并修掉的口径错误（这才是最该记住的）

1. **切片对齐**：平台 mp4 与 srt 不是同一条时间轴 → 改用整段式
2. **语言写死 zh**（Whisper + sherpa 都犯过）
3. **数字归一化缺失**：字幕「10 种人」vs ASR「二进制人」被算成 3 个错
   → `digits_to_chinese()`
4. **`for _ in range` 的 `_` 被 `segs, _ = transcribe()` 覆盖** → CER 恒 -100%
5. **artifact 未限定 run-id** → 旧数据集结果混入新表
6. **时长吻合 ≠ 内容对齐** → 新增 `validate_dataset.py`

## 局限（别拿这些数字当普适结论）

* **只有两类领域**：理工科讲课、英文讲课。缺人文艺术、快语速、方言口音、
  背景音乐 —— 这些才拉开差距。
* **朗读式讲课**语音难度低于口语/会议/电话。FunASR 官方基准（口语/会议：
  SenseVoice 7.81% vs whisper.cpp base 31.33%）与本表方向一致，但**差距幅度
  因领域而异**，不能直接对比。
* 参考是**平台官方字幕**，非独立人工转写。
* 共 **9 段**，样本量小，个别片段会主导结果（这次就发生了）。
