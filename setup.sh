#!/bin/bash
set -euo pipefail

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install -e .
