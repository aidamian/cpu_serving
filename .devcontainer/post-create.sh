#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/cpu_serving

npm install -g @openai/codex

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install -r requirements.txt

echo "Devcontainer setup complete."
