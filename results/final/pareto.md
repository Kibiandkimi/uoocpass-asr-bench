# ASR 基准结果（准 > 小 > 快）

- 数据集：**4 段 / 328 秒**，领域：人文艺术讲课、理工科讲课、英文讲课
- 参考可信度：human 60s、subtitle 88s、subtitle-misaligned 180s
  （`human`=人工校对；`subtitle`=平台官方字幕且已验证对齐；`subtitle-misaligned`=字幕时间轴错位、结果不可信；`asr`=模型草稿=循环论证）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（micro：总编辑距离/总参考长度，越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 | 段/秒 |
| --- | --- | --- | --- | --- | --- | --- |
| ★ ⚠ | sherpa-onnx SenseVoice-Small | 8.10% | 1164 | 17.44x | 0.8s | 0/0s |
| ★ ⚠ | sherpa-onnx paraformer-zh (2023-09) | 16.15% | 274 | 20.73x | 1.2s | 0/0s |
| ★ ⚠ | sherpa-onnx paraformer-zh (2024-03) | 22.93% | 1043 | 23.61x | 1.4s | 0/0s |
| ★ ⚠ | whisper.cpp base (q5_0) | 26.26% | 63 | 11.13x | 0.0s | 0/0s |
|   ⚠ | sherpa-onnx Zipformer-CTC-zh | 56.70% | 391 | 17.88x | 3.0s | 0/0s |
|   ⚠ | whisper.cpp small (q5_0) | 60.24% | 187 | 4.04x | 0.0s | 0/0s |
| ★ ⚠ | faster-whisper base/int8 | 62.69% | 400 | 21.21x | 1.9s | 0/0s |
|   ⚠ | faster-whisper small/int8 | 64.11% | 739 | 8.12x | 3.2s | 0/0s |
|   ⚠ | vosk big-cn | 64.90% | 2078 | 7.70x | 4.1s | 0/0s |
| ★ ⚠ | faster-whisper tiny/int8 | 65.44% | 330 | 22.27x | 1.9s | 0/0s |
| ★ ⚠ | whisper.cpp tiny (q5_0) | 67.21% | 37 | 7.99x | 0.0s | 0/0s |
|   ⚠ | vosk small-cn | 68.58% | 99 | 9.20x | 0.2s | 0/0s |

## 帕累托前沿

- **sherpa-onnx SenseVoice-Small**：CER 8.10%，1164MB，17.44x
- **sherpa-onnx paraformer-zh (2023-09)**：CER 16.15%，274MB，20.73x
- **sherpa-onnx paraformer-zh (2024-03)**：CER 22.93%，1043MB，23.61x
- **whisper.cpp base (q5_0)**：CER 26.26%，63MB，11.13x
- **faster-whisper base/int8**：CER 62.69%，400MB，21.21x
- **faster-whisper tiny/int8**：CER 65.44%，330MB，22.27x
- **whisper.cpp tiny (q5_0)**：CER 67.21%，37MB，7.99x
