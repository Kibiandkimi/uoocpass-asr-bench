# ASR 中文课程音频基准：又小又快又准

为 UOOCPass 的「视频转文字」挑选 ASR 后端。目标优先级 **准 > 小 > 快**，但要给出
**帕累托前沿**（没有别的方案能在三项上同时不差）供权衡。

## 候选

| 后端 | 说明 |
| --- | --- |
| faster-whisper (tiny/base/small/medium, int8/int8_float16/float32) | CTranslate2 + Whisper |
| whisper.cpp (ggml tiny/base/small, q5_0/q8_0) | 纯 C++，无 CT2 |
| vosk (small-cn / big-cn) | FST 解码，无 onnx/CT2 |
| sherpa-onnx (paraformer-zh, zipformer) | onnx 推理，中文原生模型 |

## 指标

- **准 (CER)**：字符错误率，`1 - Levenshtein(ref, hyp)/len(ref)`。参考转写人工校对。
- **小 (MB)**：运行期实际占用 = 包体积 + 模型体积（不含共享系统库）。
- **快 (RTF)**：实时倍率 = 音频秒数 / 转写秒数，越大越快。CPU-only，测 3 次取中位数。

## 产出

`pareto.md` + `pareto.svg`：三维帕累托前沿图。
