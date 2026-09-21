#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A_PROX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read install_dir from config, fall back to default
CONFIG_FILE="${A_PROX_ROOT}/config/default.toml"
if [ -f "$CONFIG_FILE" ]; then
    INSTALL_DIR=$(grep '^install_dir' "$CONFIG_FILE" | sed 's/.*= *"\(.*\)"/\1/' | sed 's|^~|'"${HOME}"'|')
else
    INSTALL_DIR="${HOME}/.local/share/a-prox-searxng"
fi
VENV_DIR="${INSTALL_DIR}/.venv"

echo "=== Setting up SearXNG ==="

# Check if SearXNG is already installed (skip install)
if [ -f "$VENV_DIR/bin/activate" ] && [ -d "$INSTALL_DIR/searxng" ] && [ -f "$INSTALL_DIR/settings.yml" ]; then
    echo "SearXNG is already installed at $INSTALL_DIR"
    echo "Skipping install."
    exit 0
fi

echo "SearXNG not found. Installing to $INSTALL_DIR..."

# Create directories
mkdir -p "$INSTALL_DIR"

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate venv
source "$VENV_DIR/bin/activate"

# Clone real SearXNG
echo "Cloning SearXNG repository..."
git clone --depth 1 https://github.com/searxng/searxng "$INSTALL_DIR/searxng" 2>/dev/null || {
    # If git fails, try downloading a tarball
    echo "Git clone failed, downloading release tarball..."
    curl -sL "https://github.com/searxng/searxng/archive/refs/heads/master.tar.gz" | tar -xz -C "$INSTALL_DIR"
    mv "$INSTALL_DIR/searxng-master" "$INSTALL_DIR/searxng"
}

cd "$INSTALL_DIR/searxng"

# Install SearXNG dependencies from requirements.txt
echo "Installing SearXNG dependencies..."
pip install --quiet setuptools wheel
pip install --quiet -r requirements.txt

# Also install searxng itself
pip install --quiet -e . --no-build-isolation 2>/dev/null || echo "SearXNG package install skipped, running from source..."

# Generate default settings.yml
echo "Generating settings.yml..."
cat > "$INSTALL_DIR/settings.yml" <<'SETTINGS'
use_default_settings: true

general:
  debug: false
  instance_name: "A-PROX SearXNG"

search:
  safe_search: 0
  autocomplete: ""
  default_lang: ""
  ban_time_between_queries: 0
  formats:
    - html
    - json

server:
  port: 8888
  bind_address: "127.0.0.1"
  secret_key: "a-prox-searxng-secret-key-change-in-production"
  limiter: false
  http_protocol_version: "1.1"
  image_proxy: false
  method: "GET"
  default_http_headers:
    X-Content-Type-Options: nosniff

engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false

  - name: bing
    engine: bing
    shortcut: b
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false
    search_type: auto

  - name: github
    engine: github
    shortcut: gh
    disabled: false

  - name: stackoverflow
    engine: stackoverflow
    shortcut: st
    disabled: false

  - name: reddit
    engine: reddit
    shortcut: rd
    disabled: false

  - name: startpage
    engine: startpage
    shortcut: sp
    disabled: false

  - name: arxiv
    engine: arxiv
    shortcut: arx
    disabled: false

  - name: pubmed
    engine: pubmed
    shortcut: pm
    disabled: false

# Redis is optional - SearXNG works without it
# redis_url: ""

outgoing:
  request_timeout: 5.0
  max_request_timeout: 15.0
  useragent_suffix: "A-PROX"
  max_retries: 1

# UI settings
ui:
  static_use_hash: true
  default_theme: simple
  theme_args:
    style_theme: ""
  pagination_args:
    simple_style: true
  infinite_scroll: true
  default_locale: ""

# Performance
performance:
  enabled: true
SETTINGS

# Deactivate venv
deactivate 2>/dev/null || true

echo ""
echo "=== SearXNG Setup Complete ==="
echo "  Location: $INSTALL_DIR"
echo "  Venv:     $VENV_DIR"
echo "  Settings: $INSTALL_DIR/settings.yml"
echo "  Port:     8888"
echo "  Command:  $VENV_DIR/bin/python searxng/run.py"
