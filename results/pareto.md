# ASR 基准结果（准 > 小 > 快）

- 音频：`course_zh_60s.wav`（60 秒中文课程讲课）
- 参考文本：227 字（人工校对）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 |
| --- | --- | --- | --- | --- | --- |
| ★ | faster-whisper small/int8 | 7.05% | 739 | 2.35x | 2.6s |
| ★ | faster-whisper base/int8 | 11.01% | 400 | 6.68x | 1.4s |
| ★ | faster-whisper tiny/int8 | 19.82% | 330 | 12.27x | 2.3s |
| ★ | vosk small-cn | 20.26% | 99 | 3.52x | 0.6s |

## 帕累托前沿

- **faster-whisper small/int8**：CER 7.05%，739MB，2.35x
- **faster-whisper base/int8**：CER 11.01%，400MB，6.68x
- **faster-whisper tiny/int8**：CER 19.82%，330MB，12.27x
- **vosk small-cn**：CER 20.26%，99MB，3.52x
