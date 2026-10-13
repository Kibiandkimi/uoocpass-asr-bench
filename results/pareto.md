# ASR 基准结果（准 > 小 > 快）

- 数据集：**9 段 / 1666 秒**，领域：理工科讲课、英文讲课
- 参考可信度：subtitle 1666s
  （`human`=人工校对；`subtitle`=平台官方字幕且已验证对齐；`subtitle-misaligned`=字幕时间轴错位、结果不可信；`asr`=模型草稿=循环论证）
- ★ = 三维帕累托前沿上的方案
- CER = 字符错误率（micro：总编辑距离/总参考长度，越低越准）；size = 包 + 模型实际占用；rtf = 实时倍率（越大越快）

| ★ | 方案 | CER ↓ | 体积 MB ↓ | RTF ↑ | 加载 | 段/秒 |
| --- | --- | --- | --- | --- | --- | --- |
| | sherpa-onnx paraformer-zh (2023-09) | — | 270 | — | — | missing: sherpa-onnx 未安装：No module named 'sherpa_ |

## 帕累托前沿

（本轮没有成功的方案）
