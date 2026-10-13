# ASR 基准结果（准 > 小 > 快）

- 音频：`course_zh_60s.wav`（60 秒中文课程讲课）
- 参考文本：227 字（人工校对）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 |
| --- | --- | --- | --- | --- | --- |
|   | faster-whisper small/int8 | 7.05% | 739 | 2.35x | 2.6s |
| ★ | sherpa-onnx paraformer-zh | 7.05% | 274 | 20.28x | 1.2s |
|   | faster-whisper medium/int8 | 8.81% | 1785 | 1.21x | 9.4s |
|   | faster-whisper base/int8 | 10.13% | 400 | 10.95x | 2.8s |
|   | vosk big-cn | 11.45% | 2078 | 4.23x | 4.9s |
|   | faster-whisper tiny/int8 | 18.50% | 330 | 15.32x | 1.8s |
| ★ | vosk small-cn | 20.26% | 99 | 3.52x | 0.6s |

## 帕累托前沿

- **sherpa-onnx paraformer-zh**：CER 7.05%，274MB，20.28x
- **vosk small-cn**：CER 20.26%，99MB，3.52x
