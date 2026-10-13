# 已修的 bug（都影响过结论，记下来防止再犯）

## 1. `for _ in range(...)` 的 `_` 被覆盖 ⚠️ 最严重

```python
for _ in range(repeats):
    segs, _ = model.transcribe(...)   # ← 把循环变量 _ 覆盖成 TranscriptionInfo
    if _ == 0:                        # ← 永远 False
        per_clip[clip.id] = ...       # ← 从没执行
```

后果：`per_clip` 恒空 → micro-CER 恒 `-1.0` → 报告里 faster-whisper 全系列
显示 **CER -100%**，而 RTF 是「总时长 ÷ 有效耗时」，数字看着正常，很容易被骗过去。

修法：循环变量改名 `rep`，解包用 `_info`。

## 2. `run_candidate` 漏拷字段

子进程 JSON 里有 `per_clip` / `clips` / `clip_seconds` / `weakest_verified`，
父进程 `run_candidate` 只拷了 `cer` / `rtf` / `load_s` / `text` / `size_mb`。
于是每段明细全丢，报告只剩一个 micro-CER —— **看不出模型是在哪类音频上崩的**。
（这也是为什么 vosk 在单段 60s 时看着 20% 不错，换到 4 段变 68% 才发现问题。）

## 3. artifact 没限定 run-id

`actions/download-artifact` 不带 `run-id` 会把仓库里**历次**的 artifact 全捞下来，
旧数据集（单段 60 秒）的结果混进新结果，出了一张自相矛盾的表
（paraformer 7.05% 与 SenseVoice 8.10% 并列，前者其实是旧数据）。

修法：`run-id: ${{ github.run_id }}` + `github-token`。

## 4. 子进程 JSON 解析只取最后一行

`json.loads(proc.stdout.splitlines()[-1])` —— 但库会往 stdout 打警告/进度，
真正的 JSON 被夹在中间。改成**从后往前找第一个能解析的 dict**。

## 5. 数据集对齐（**未修完**）

平台给的 `video_url`(mp4) 与 `subtitle`(srt) **不是同一条时间轴**：
视频带片头、SRT 从讲稿起点计时。实测「穷举法」那节：字幕 62s 讲「国王的婚姻/
旅行商问题」，而 mp4 60s 处音频在讲「3x+y+z=100」。扫 ±180s 固定偏移也救不回来。

所以现在 4 段里 **180 秒是 `subtitle-misaligned`**，这部分 CER 不可信（会虚高）。
**修法**：不做切片、整段视频配整段字幕，CER 在完整视频上算 —— 从根上消除对齐问题。

## 6. bash + curl 的模型下载在 runner 上「秒失败且无输出」

同一个 runner 直接敲 `curl` 下载 190MB 完全正常，脚本里却立即失败，
`bash -x` 的跟踪也没落到日志里。换成 Python 下载器（`scripts/fetch_models.py`）后
一切正常，且每个 URL 的失败原因都会打进日志。**不要再回去用 bash 拼 curl。**
