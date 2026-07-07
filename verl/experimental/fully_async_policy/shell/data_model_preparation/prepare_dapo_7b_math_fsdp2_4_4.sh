#!/usr/bin/env bash
# Prepare data and model for:
#   verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh
#
# Produces the following layout (matches RAY_DATA_HOME defaults in that script):
#   ${RAY_DATA_HOME}/data/dapo-math-17k.parquet
#   ${RAY_DATA_HOME}/data/aime-2024.parquet
#   ${RAY_DATA_HOME}/models/Qwen2.5-Math-7B/
#
# Usage:
#   bash prepare.sh
#
# Optional env vars:
#   RAY_DATA_HOME   Root directory for data and models (default: $HOME/verl)
#   OVERWRITE       Set to 1 to re-download files that already exist
#   HF_TOKEN        Hugging Face access token (if needed for gated models)
#
# After this script completes, launch training with:
#   bash /workspace/verl/verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh

set -euo pipefail

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
OVERWRITE="${OVERWRITE:-0}"

DATA_DIR="${RAY_DATA_HOME}/data"
MODEL_DIR="${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"
TRAIN_FILE="${DATA_DIR}/dapo-math-17k.parquet"
TEST_FILE="${DATA_DIR}/aime-2024.parquet"

TRAIN_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
TEST_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
MODEL_ID="Qwen/Qwen2.5-Math-7B"

# ── helpers ───────────────────────────────────────────────────────────────────

log() { echo "[prepare] $*"; }

download_file() {
    local url="$1" dest="$2"
    if command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "${dest}.partial" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "${dest}.partial" "${url}"
    else
        echo "ERROR: neither wget nor curl found." >&2; exit 1
    fi
    mv "${dest}.partial" "${dest}"
}

# ── directories ───────────────────────────────────────────────────────────────

mkdir -p "${DATA_DIR}" "${RAY_DATA_HOME}/models"

# ── datasets ──────────────────────────────────────────────────────────────────

if [[ ! -f "${TRAIN_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
    log "Downloading training data -> ${TRAIN_FILE}"
    download_file "${TRAIN_URL}" "${TRAIN_FILE}"
else
    log "Skip (exists): ${TRAIN_FILE}"
fi

if [[ ! -f "${TEST_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
    log "Downloading validation data -> ${TEST_FILE}"
    download_file "${TEST_URL}" "${TEST_FILE}"
else
    log "Skip (exists): ${TEST_FILE}"
fi

# ── model ─────────────────────────────────────────────────────────────────────

if [[ ! -f "${MODEL_DIR}/config.json" || "${OVERWRITE}" -eq 1 ]]; then
    log "Downloading model ${MODEL_ID} -> ${MODEL_DIR}"
    if command -v hf >/dev/null 2>&1; then
        HF_CMD="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CMD="huggingface-cli"
    else
        echo "ERROR: hf (huggingface_hub CLI) not found. Install with:" >&2
        echo "  pip install -U 'huggingface_hub[cli]'" >&2
        exit 1
    fi
    "${HF_CMD}" download "${MODEL_ID}" \
        --repo-type model \
        --local-dir "${MODEL_DIR}" \
        ${HF_TOKEN:+--token "${HF_TOKEN}"}
else
    log "Skip (exists): ${MODEL_DIR}"
fi

# ── patch config.json ─────────────────────────────────────────────────────────
# The training script overrides max_position_embeddings=32768 via Hydra, and
# the comment at line 13 of dapo_7b_math_fsdp2_4_4.sh explicitly requires this.

log "Patching ${MODEL_DIR}/config.json: max_position_embeddings -> 32768"
python3 - "${MODEL_DIR}/config.json" <<'EOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)
if cfg.get("max_position_embeddings") != 32768:
    cfg["max_position_embeddings"] = 32768
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"  updated {path}")
else:
    print(f"  already 32768, no change needed")
EOF

# ── summary ───────────────────────────────────────────────────────────────────

log "Done. Asset locations:"
log "  TRAIN_FILE  = ${TRAIN_FILE}"
log "  TEST_FILE   = ${TEST_FILE}"
log "  MODEL_PATH  = ${MODEL_DIR}"
log ""
log "To start training:"
log "  bash /workspace/verl/verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh"
