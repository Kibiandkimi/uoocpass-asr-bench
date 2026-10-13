#!/usr/bin/env bash
# 下载各后端的模型到 models/。已存在就跳过，省得重复下几百 MB。
set -u
cd "$(dirname "$0")/.."
mkdir -p models

get() {  # get <url> <目标路径>
  local url="$1" dest="$2"
  [ -e "$dest" ] && { echo "  ✓ 已有 $dest"; return; }
  echo "  ↓ $url"
  if ! curl -fL --retry 3 -o "$dest.part" "$url"; then
    echo "  ✗ 下载失败 $url"; rm -f "$dest.part"; return; fi
  mv "$dest.part" "$dest"
}

echo "== vosk 中文 =="
get https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip models/vosk-small-cn.zip
[ -d models/vosk-model-small-cn-0.22 ] || unzip -q models/vosk-small-cn.zip -d models/
get https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip models/vosk-big-cn.zip
[ -d models/vosk-model-cn-0.22 ] || unzip -q models/vosk-big-cn.zip -d models/

echo "== whisper.cpp ggml =="
get https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_0.bin models/ggml-tiny-q5_0.bin
get https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_0.bin models/ggml-base-q5_0.bin
get https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_0.bin models/ggml-small-q5_0.bin

echo "== sherpa-onnx paraformer-zh =="
get https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 models/paraformer-zh.tar.bz2
[ -d models/sherpa-onnx-paraformer-zh-2023-09-14 ] || tar xjf models/paraformer-zh.tar.bz2 -C models/

echo "== faster-whisper 模型（交给 HF 缓存）=="
echo "  设 HF_HOME=models/hf，首次 transcribe 会自动下载"
echo "完成"
