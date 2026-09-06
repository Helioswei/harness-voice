#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "尚未安装依赖，请先运行: bash scripts/install.sh"
    exit 1
fi

exec .venv/bin/python -m voice.main
