#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"

echo "=== Starting A-PROX (Hardware-Optimized LLM Middleware) ==="

# Check and download embedding model if needed
if [ ! -f "$DIR/models/bge-small-en-v1.5-int8.onnx" ]; then
    echo "Downloading embedding models first..."
    "$DIR/scripts/download_model.sh"
fi

# Check and download SearXNG if needed (enabled by default)
SEARXNG_VENV="$DIR/searxng/.venv/bin/activate"
if [ ! -f "$SEARXNG_VENV" ]; then
    echo ""
    echo "SearXNG not found. Downloading and installing..."
    "$DIR/scripts/download_searxng.sh"
fi

# Rebuild before launching
echo "Building A-PROX..."
cargo build --release

# Run A-PROX binary
exec "$DIR/target/release/a-prox" --config "$DIR/config/default.toml" "$@"
