#!/usr/bin/env bash
# 下载各后端模型到 models/。已存在就跳过。
#
# 两个坑，这里都处理了：
#   1. whisper.cpp 的 ggml 文件名**不能猜** —— ``ggml-tiny-q5_0.bin`` 在 HF 上
#      其实是 404；哪些量化后缀真实存在得问 API。
#   2. 下载失败**不能静默** —— 上一版里 ``fetch_models.sh || true`` 把 404 吞了，
#      结果三个 wcpp 候选全部以「缺模型」告终，白跑一轮 CI。
set -u
cd "$(dirname "$0")/.."
mkdir -p models

FAILED=()

get() {  # get <url> <目标路径> [可选备用URL...]（备用 url 需自带目标，见用法）
  local dest="$2"; shift 2
  [ -e "$dest" ] && { echo "  ✓ 已有 $(basename "$dest")"; return 0; }
  local url
  for url in "$@"; do
    echo "  ↓ $(basename "$dest") ← $url"
    if curl -fL --retry 3 --retry-delay 2 --max-time 900 -o "$dest.part" "$url"; then
      mv "$dest.part" "$dest"; return 0
    fi
    rm -f "$dest.part"
  done
  echo "  ✗ 全部镜像都失败：$dest"
  FAILED+=("$dest")
  return 1
}

# 先问 HF API 有哪些 ggml 变体，挑一个真实存在的（避免猜文件名撞 404）
hf_pick() {  # hf_pick <repo> <基础名>
  curl -fsSL --max-time 60 "https://huggingface.co/api/models/$1" 2>/dev/null \
    | python3 -c "
import json, sys
base = sys.argv[1]
try:
    files = [f['rfilename'] for f in json.load(sys.stdin).get('siblings', [])]
except Exception:
    sys.exit(0)
for suffix in ('-q5_0.bin', '-q5_1.bin', '-q8_0.bin', '.bin'):
    for f in files:
        if f == base + suffix or f.endswith('/' + base + suffix):
            print(f); sys.exit(0)
" "$2"
}

echo "== vosk 中文 =="
get https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip models/vosk-small-cn.zip
[ -d models/vosk-model-small-cn-0.22 ] || unzip -q models/vosk-small-cn.zip -d models/ 2>/dev/null || true
get https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip models/vosk-big-cn.zip
[ -d models/vosk-model-cn-0.22 ] || unzip -q models/vosk-big-cn.zip -d models/ 2>/dev/null || true

echo "== whisper.cpp ggml =="
HF=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
for size in tiny base small; do
  base="ggml-$size"
  picked=$(hf_pick ggerganov/whisper.cpp "$base" || true)
  if [ -n "$picked" ]; then
    echo "  · $base → $picked（API 给出）"
    get "$HF/$picked" "models/$(basename "$picked")" || true
  else
    echo "  · $base：API 无响应，试常见量化后缀"
    for suffix in q5_0 q5_1 q8_0; do
      get "$HF/$base-$suffix.bin" "models/$base-$suffix.bin" && break
    done
  fi
done
# bench.py 统一按 ggml-<size>-q5_0.bin 找文件 → 用软链兼容实际下到的文件名
for size in tiny base small; do
  for f in models/ggml-$size-q5_*.bin models/ggml-$size-q8_*.bin models/ggml-$size.bin; do
    [ -f "$f" ] || continue
    ln -sf "$(basename "$f")" "models/ggml-$size-q5_0.bin" 2>/dev/null && break
  done
done

echo "== sherpa-onnx paraformer-zh =="
get https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 \
    models/paraformer-zh.tar.bz2 || true
[ -d models/sherpa-onnx-paraformer-zh-2023-09-14 ] \
  || tar xjf models/paraformer-zh.tar.bz2 -C models/ 2>/dev/null || true

echo "== faster-whisper =="
echo "  设 HF_HOME=models/hf，首次 transcribe 会自动下载"

echo
echo "== 小结 =="
du -sh models/* 2>/dev/null | sort -rh | head -14
if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "!! ${#FAILED[@]} 个模型没下到："
  printf '   %s\n' "${FAILED[@]}"
  exit 1
fi
echo "全部就绪"
