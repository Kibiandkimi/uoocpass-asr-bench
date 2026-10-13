# ASR 基准结果（准 > 小 > 快）

- 数据集：**9 段 / 1666 秒**，领域：理工科讲课、英文讲课
- 参考可信度：subtitle 1666s
  （`human`=人工校对；`subtitle`=平台官方字幕且已验证对齐；`subtitle-misaligned`=字幕时间轴错位、结果不可信；`asr`=模型草稿=循环论证）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（micro：总编辑距离/总参考长度，越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 | 段/秒 |
| --- | --- | --- | --- | --- | --- | --- |
| ★ | faster-whisper small/int8 | 4.20% | 739 | 4.05x | 6.3s | 9/1703s |
| ★ | faster-whisper base/int8 | 4.79% | 400 | 11.23x | 3.3s | 9/1703s |
| ★ | whisper.cpp base (q5_0) | 4.81% | 63 | 5.68x | 0.0s | 9/1703s |
| ★ | whisper.cpp tiny (q5_0) | 7.30% | 37 | 17.61x | 0.0s | 9/1703s |
|   | faster-whisper tiny/int8 | 9.39% | 330 | 16.07x | 2.4s | 9/1703s |
|   | sherpa-onnx SenseVoice-Small | 15.80% | 1164 | 10.57x | 0.6s | 9/1703s |
|   | sherpa-onnx paraformer-zh (2023-09) | 27.48% | 274 | 11.30x | 1.1s | 9/1703s |
|   | sherpa-onnx paraformer-zh (2024-03) | 29.26% | 1043 | 12.94x | 1.3s | 9/1703s |
|   | vosk big-cn | 64.13% | 2078 | 2.51x | 3.6s | 9/1703s |
|   | vosk small-cn | 66.31% | 99 | 4.78x | 0.2s | 9/1703s |

## 帕累托前沿

- **faster-whisper small/int8**：CER 4.20%，739MB，4.05x
- **faster-whisper base/int8**：CER 4.79%，400MB，11.23x
- **whisper.cpp base (q5_0)**：CER 4.81%，63MB，5.68x
- **whisper.cpp tiny (q5_0)**：CER 7.30%，37MB，17.61x
