# 已知问题

## CI 上模型下载失败（未解决）

`fetch_models.sh` 在 GitHub runner 上失败，但**同一 runner 直接 curl 是成功的**：

```
curl -fL .../ggml-small-q5_1.bin   → OK 190085487 bytes   ✅
curl -fL .../vosk-model-small-cn   → OK 43898754 bytes    ✅
curl -fsSL .../api/models/...      → OK 4035 bytes        ✅
bash scripts/fetch_models.sh       → 立即失败，且无输出    ❌
```

脚本第 4 段用 `bash -x` 抓的跟踪也没能落到 CI 日志里（被 step 自身的失败截断）。
**排查方向**（下次接着查）：

1. `get()` 里 `local url; for url in "$@"` 在 `set -u` 下对空参的处理；
2. 成功下载后的 `mv "$dest.part" "$dest"` 与 `du` 段是否触发了 `set -e`；
3. 脚本中间那批 `[ -d ... ] || unzip ... || true` 的返回值链；
4. 在 CI 里加 `set -x` 到脚本内容（而不是调用时 `bash -x`），并把 stdout/stderr
   重定向到文件后**无论如何都 cat 出来**（用 `if: always()`）。

在此之前，whisper.cpp / vosk-big / sherpa 的 CI 结果不可靠；
但 **faster-whisper 系列和 vosk-small 是可信的**（模型的下载路径与上面无关）。
