#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/bin/python ]; then
    echo "Run bash setup_unix.sh first"
    exit 1
fi
.venv/bin/python -m streamlit run app.py
