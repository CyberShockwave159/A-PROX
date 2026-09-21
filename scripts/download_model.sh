#!/usr/bin/env bash
set -e

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODEL_DIR"

echo "=== Downloading Quantized CPU ONNX Model (bge-small-en-v1.5) ==="
MODEL_FILE="$MODEL_DIR/bge-small-en-v1.5-int8.onnx"
TOKENIZER_FILE="$MODEL_DIR/tokenizer.json"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model_quantized.onnx (~67 MB)..."
    curl -L --progress-bar -o "$MODEL_FILE" \
        "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx"
    echo "Saved to $MODEL_FILE"
else
    echo "Model already exists at $MODEL_FILE"
fi

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "Downloading tokenizer.json (~2 MB)..."
    curl -L --progress-bar -o "$TOKENIZER_FILE" \
        "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json"
    echo "Saved to $TOKENIZER_FILE"
else
    echo "Tokenizer already exists at $TOKENIZER_FILE"
fi

echo "=== CPU Embedding Model Setup Complete! ==="
