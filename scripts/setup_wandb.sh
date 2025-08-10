#!/usr/bin/env bash
# Usage: ./scripts/setup_wandb.sh <API_KEY> [ENTITY] [PROJECT]
set -euo pipefail

API_KEY=${1:-}
ENTITY=${2:-}
PROJECT=${3:-}

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 <API_KEY> [ENTITY] [PROJECT]" >&2
  exit 1
fi

# Append to ~/.bashrc for persistence
{
  echo "export WANDB_API_KEY=$API_KEY"
  if [[ -n "$ENTITY" ]]; then
    echo "export WANDB_ENTITY=$ENTITY"
  fi
  if [[ -n "$PROJECT" ]]; then
    echo "export WANDB_PROJECT=$PROJECT"
  fi
} >> "$HOME/.bashrc"

echo "Wrote WANDB env vars to ~/.bashrc. Reload your shell or run: source ~/.bashrc"


