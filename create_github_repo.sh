#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${REPO_NAME:-BRRH-ZoeDepth}"
VISIBILITY="${VISIBILITY:-public}"

if [[ -z "${GITHUB_OWNER:-}" ]]; then
  echo "Set GITHUB_OWNER first, for example:"
  echo "  export GITHUB_OWNER=your-github-username"
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required."
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
fi

git add .
git commit -m "Release BRRH-ZoeDepth code and paper artifacts" || true

if command -v gh >/dev/null 2>&1; then
  gh repo create "${GITHUB_OWNER}/${REPO_NAME}" "--${VISIBILITY}" --source . --remote origin --push
else
  echo "GitHub CLI (gh) is not installed."
  echo "Create the repository manually at https://github.com/new with name: ${REPO_NAME}"
  echo "Then run:"
  echo "  git remote add origin https://github.com/${GITHUB_OWNER}/${REPO_NAME}.git"
  echo "  git branch -M main"
  echo "  git push -u origin main"
fi
