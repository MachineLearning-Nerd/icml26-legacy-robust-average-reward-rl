#!/usr/bin/env bash
# Fixed run command (identical on every experiment node). Boots uv, syncs the
# locked env from pyproject.toml/uv.lock, then runs the full claim-verification
# CLI which writes EVAL.md + raw artifacts under .openresearch/artifacts/.
set -euo pipefail

# --- bootstrap uv (no system pip) -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- reproducible env from the lockfile --------------------------------------
uv sync --frozen --no-progress

# --- run every claim verifier -----------------------------------------------
uv run python -m rhi_repro.cli
