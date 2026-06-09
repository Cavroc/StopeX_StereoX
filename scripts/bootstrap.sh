#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python 3 is required but was not found on PATH." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cat <<'EOF'
Environment ready.

Next steps:
  source .venv/bin/activate
  python stereonet_app.py

Then open http://127.0.0.1:8050
EOF

