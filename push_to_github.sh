#!/bin/bash
# Replace the contents of the GitHub repo with the new project, in one commit. Old history is kept
# (old files remain recoverable from earlier commits). Usage:
#   bash push_to_github.sh [zip] [remote-url]
# Authentication: be logged in already (`gh auth login`) or use a token URL for the remote.
set -euo pipefail
ZIP=$(realpath "${1:-Beyond-Retrieval-final.zip}")
REMOTE=${2:-https://github.com/Gyanankur23/Beyond-Retrieval-Evidence-Grounded-Agentic-AI-Systems.git}
[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }
git config user.name  >/dev/null 2>&1 || { echo "Run: git config --global user.name 'Your Name'"; exit 1; }
git config user.email >/dev/null 2>&1 || { echo "Run: git config --global user.email you@example.com"; exit 1; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
git clone -q "$REMOTE" "$work/repo"
cd "$work/repo"
BRANCH=$(git symbolic-ref --short HEAD)
git rm -r -q --ignore-unmatch . && git clean -fdxq            # remove every old tracked/untracked file
unzip -q "$ZIP" -d "$work/new"
cp -a "$work"/new/*/. .                                        # zip has one top-level folder
git add -A
git commit -q -m "Reframe project: version-aware retrieval measurement study (real data, reproducible, no external APIs)"
git push origin "$BRANCH"
echo "Done. Pushed $(git rev-list --count HEAD) commits to $REMOTE ($BRANCH)."
