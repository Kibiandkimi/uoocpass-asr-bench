# ASR 基准结果（准 > 小 > 快）

- 数据集：**9 段 / 1666 秒**，领域：理工科讲课、英文讲课
- ⚠ 含中英混说片段：en-1911877932-2（单语模型处理 code-switching 会掉点，不宜与单语片段混算）
- 参考可信度：subtitle 1666s
  （`human`=人工校对；`subtitle`=平台官方字幕且已验证对齐；`subtitle-misaligned`=字幕时间轴错位、结果不可信；`asr`=模型草稿=循环论证）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（micro：总编辑距离/总参考长度，越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER(micro) ↓ | CER(macro) ↓ | 体积 MB ↓ | RTF ↑ | 加载 | 段/秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ★ | whisper.cpp base (q5_0) | 3.71% | 3.69% | 63 | 5.75x | 0.0s | 9/1703s |
|   | whisper.cpp small (q5_0) | 3.82% | 4.12% | 187 | 4.00x | 0.0s | 9/1703s |
| ★ | faster-whisper base/int8 | 3.87% | 4.12% | 400 | 10.40x | 4.0s | 9/1703s |
|   | faster-whisper small/int8 | 4.44% | 7.06% | 739 | 4.06x | 3.9s | 9/1703s |
| ★ | whisper.cpp tiny (q5_0) | 7.29% | 9.82% | 37 | 8.91x | 0.0s | 9/1703s |
| ★ | faster-whisper tiny/int8 | 8.52% | 9.17% | 330 | 15.40x | 2.1s | 9/1703s |
|   | sherpa-onnx SenseVoice-Small | 13.74% | 8.67% | 1164 | 9.46x | 0.7s | 9/1703s |
| ★ | sherpa-onnx paraformer-zh (2023-09) | 25.28% | 18.95% | 274 | 14.27x | 0.8s | 9/1703s |
|   | sherpa-onnx paraformer-zh (2024-03) | 29.91% | 21.42% | 1043 | 14.65x | 1.2s | 9/1703s |
|   | vosk big-cn | 63.41% | 38.23% | 2078 | 2.33x | 4.6s | 9/1703s |
|   | vosk small-cn | 65.75% | 42.18% | 99 | 4.64x | 0.2s | 9/1703s |

## 帕累托前沿

- **whisper.cpp base (q5_0)**：CER 3.71%，63MB，5.75x
- **faster-whisper base/int8**：CER 3.87%，400MB，10.40x
- **whisper.cpp tiny (q5_0)**：CER 7.29%，37MB，8.91x
- **faster-whisper tiny/int8**：CER 8.52%，330MB，15.40x
- **sherpa-onnx paraformer-zh (2023-09)**：CER 25.28%，274MB，14.27x
