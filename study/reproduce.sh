#!/bin/sh
set -e
cd "$(dirname "$0")"
python scripts/build_sections.py
python experiments/run_study.py
python scripts/make_readme.py
if command -v pdflatex >/dev/null; then sh paper/build.sh; else echo "pdflatex not found: skipped paper build"; fi
