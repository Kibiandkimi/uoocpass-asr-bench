# 状态与下一步

## 已完成 ✅

### 数据集（阻断项已解）

**整段式：9 段 / 27.8 分钟 / 2 个语种 / 理工+英文两类**（`audio/dataset.json`）

整段式的关键好处：**没有切片对齐问题**。平台给的 `video_url`(mp4) 与
`subtitle`(srt) 不是同一条时间轴（视频带片头），按时间戳切片段必然错位。
但**整段**时 SRT 末条时间戳与音频时长实测吻合（差 1~7 秒），所以整段可比。

切片版留在 `audio/dataset_sliced.json`，其中 180 秒标了 `subtitle-misaligned`。

### 已修的 bug（详见 NOTES.md）

1. `for _ in range(...)` 的 `_` 被 `segs, _ = transcribe(...)` 覆盖 → per_clip 恒空、CER 恒 -100%
2. `run_candidate` 漏拷 per_clip / clips / clip_seconds / weakest_verified
3. artifact 未限定 run-id → 旧数据集结果混入
4. 子进程 JSON 只取最后一行（被库的 stdout 警告挤掉）
5. bash+curl 下载器在 runner 上秒失败 → 换 Python（错误可见 + 镜像）
6. 语言写死 `zh` → 英文片段被强行按中文解码，Whisper 系一律 CER 100%

## 待办

### P1：结论要复核 —— 整段结果推翻了公开基准

整段式 27.8 分钟的真实课程音频上：

| 候选 | CER | 体积 | RTF |
| --- | --- | --- | --- |
| faster-whisper small/int8 | 4.20% | 739MB | 4.05x |
| faster-whisper base/int8 | 4.79% | 400MB | 11.23x |
| **whisper.cpp base (q5_0)** | **4.81%** | **63MB** | 5.68x |
| **whisper.cpp tiny (q5_0)** | **7.30%** | **37MB** | 17.61x |
| SenseVoice-Small | 15.80% | 1164MB | 10.57x |
| paraformer-zh (2023-09) | 27.48% | 274MB | 11.30x |
| vosk small-cn | 66.31% | 99MB | 4.78x |

**这跟 FunASR 官方基准（SenseVoice 7.81% vs whisper.cpp base 31.33%）完全相反。**
可能的原因（需要逐个排除）：

1. **领域不匹配**：FunASR 的 184 段是**口语/会议/电话**音频，语音识别难度高；
   我们的样本是**朗读式讲课**，本就更简单 —— 简单样本上模型差距会压缩。
2. **`normalize_zh` 口径不同**：官方用 micro-CER + normalize_zh（含数字规整等），
   我只做了「剥标点 + 繁转简」。数字/英文混排的处理差异会显著影响 CER。
3. **SenseVoice 用法可能不对**：`from_sense_voice(..., language="zh")` ——
   sherpa 的 SenseVoice 需要正确的 `use_itn` / `language` 组合，配错会掉点。
4. **分段策略**：官方用 fsmn-vad 先切段再识别；我们整段直接喂，
   SenseVoice 是编码器+CTC，长音频可能要分段才发挥。
5. **参数量与领域适配**：whisper.cpp 用 q5_0 量化，居然打赢 SenseVoice —— 
   在朗读样本上不是不可能，但需要独立验证。

**结论**：在把 1~4 排除之前，**不要**据此换掉 UOOCPass 的 ASR 后端。
优先做 3（查 sherpa 的 SenseVoice 正确用法）和 2（对齐 normalize 口径）。

### P2：数据集还要扩

- 目标 **≥ 30 分钟 / ≥ 10 段 / ≥ 4 领域**（现在 27.8 分钟 / 9 段 / 2 领域）
- 缺：人文艺术（艺术美学 0 字幕，只能靠 ASR 转，**不能当参考**）
- 缺：快语速 / 带口音 / 有背景音乐 的样本（这些才拉开差距）

### P3：sherpa-zipformer-ctc 在整段数据集上失败

`bench (sherpa-zipformer-ctc): failure` —— 需要看日志。它其实是 CTC 模型，
长音频可能要配 VAD 分段。

---

## 追加：两个「口径」问题（都会系统性冤枉模型）

### A. 语言写死（**已修**）

Whisper 与 sherpa 两个分支都把 `language` 写死成 `zh`。后果：英文片段被强行
按中文解码。实测各段 CER 一看就明白：

```
whisper 系       [4 3 3 4 12 4 3 | 3 9 | 0]   ← 第 8 段（英文）正常
sherpa SenseVoice[2 4 3 3 7 3 2 | 67 | 1]      ← 第 8 段崩到 67%
```

SenseVoice 前 7 段（中文）**2~7%，其实是最强的**；只有英文段被语言参数坑了。
修法：语言按片段指定；混合语种时 sherpa 取 `"auto"`、whisper.cpp 取 `auto`。

### B. 数字归一化缺失（**已修**）

官方基准用 `normalize_zh`，我只做了「剥标点 + 繁转简」。而字幕原文与 ASR 输出的
数字形态经常不一致：

```
ref: 世界上一共有 10 种人       ASR: 世界上一共有二进制人
```

字符级编辑距离把「10」对「二进制」算 3 个错误，**其实语义是同一处**。
修法：`digits_to_chinese()` —— 10/100/1000/10000 用中文读法（十/一百/一千/一万），
其余逐位映射。

### 待重跑

这两个修完必须**重跑全部候选**才能得到最终数字。之前那版结论
（whisper.cpp base 4.81% 打赢 SenseVoice 15.80%）**不可采信**。
