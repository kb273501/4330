#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# scikit-learn ships wheels for 3.10-3.13 only, so refuse anything newer.
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "No supported Python found (need 3.10-3.13)."
    echo "Install one, for example: brew install python@3.12"
    exit 1
fi

echo "Using $PY ($("$PY" --version 2>&1))"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Run: bash start_unix.sh"
