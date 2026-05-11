#!/bin/bash
cd "$(dirname "$0")/backend"
source ../.venv/bin/activate
python3 -m pytest tests/ -v --tb=line 2>&1 | tail -20
