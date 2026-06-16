#!/usr/bin/env bash
# Download and prepare data + model for:
#   verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh
#
# Usage:
#   bash /verl_dev/fully_async/prepare_geo3k_qwen25vl_7b_megatron_4_4.sh
#
# Optional environment variables:
#   VERL_ROOT            Path to verl repo (default: /workspace/verl)
#   OVERWRITE=1          Re-download / re-preprocess even if outputs exist
#   SKIP_DATA=1          Skip dataset download and preprocessing
#   SKIP_MODEL=1         Skip model download
#   GEO3K_DIR            Output dir for train/test parquet (default: $HOME/data/geo3k)
#   HF_MODEL_PATH        Local model dir (default: ${RAY_DATA_HOME:-$HOME}/models/Qwen2.5-VL-7B-Instruct)
#   RAW_DATASET_DIR      Cache dir for raw Geometry3k dataset
#   HF_TOKEN             Hugging Face token (if needed for gated assets)

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="${VERL_ROOT:-/workspace/verl}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-${VERL_ROOT}/verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh}"

OVERWRITE=${OVERWRITE:-0}
SKIP_DATA=${SKIP_DATA:-0}
SKIP_MODEL=${SKIP_MODEL:-0}

GEO3K_DIR="${GEO3K_DIR:-${HOME}/data/geo3k}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${RAY_DATA_HOME:-${HOME}}/models/Qwen2.5-VL-7B-Instruct}"
RAW_DATASET_DIR="${RAW_DATASET_DIR:-${HOME}/downloads/datasets/hiyouga_geometry3k}"

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ID="hiyouga/geometry3k"

TRAIN_FILE="${GEO3K_DIR}/train.parquet"
TEST_FILE="${GEO3K_DIR}/test.parquet"

if ! python3 - <<'EOF'
import verl  # noqa: F401
EOF
then
    echo "verl is not importable. Install it first, e.g.:"
    echo "  cd ${VERL_ROOT} && pip install -e ."
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "hf not found. Install with: pip install -U huggingface_hub"
    exit 1
fi

############################ Download raw dataset ############################

if [ "${SKIP_DATA}" -eq 0 ]; then
    mkdir -p "${GEO3K_DIR}" "$(dirname "${RAW_DATASET_DIR}")"

    if [ ! -d "${RAW_DATASET_DIR}" ] || [ "${OVERWRITE}" -eq 1 ]; then
        echo "Downloading raw dataset ${DATASET_ID} to ${RAW_DATASET_DIR}..."
        hf download "${DATASET_ID}" \
            --repo-type dataset \
            --local-dir "${RAW_DATASET_DIR}"
    else
        echo "Raw dataset already exists at ${RAW_DATASET_DIR}, skipping download."
    fi

    ############################ Preprocess dataset ############################

    if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
        echo "Preprocessing Geometry3k to parquet in ${GEO3K_DIR}..."
        python3 "${VERL_ROOT}/examples/data_preprocess/geo3k.py" \
            --local_dataset_path "${RAW_DATASET_DIR}" \
            --local_save_dir "${GEO3K_DIR}"
    else
        echo "Preprocessed data already exists:"
        echo "  ${TRAIN_FILE}"
        echo "  ${TEST_FILE}"
    fi
else
    echo "SKIP_DATA=1, skipping dataset download and preprocessing."
fi

############################ Download model ############################

if [ "${SKIP_MODEL}" -eq 0 ]; then
    mkdir -p "$(dirname "${HF_MODEL_PATH}")"

    if [ ! -f "${HF_MODEL_PATH}/config.json" ] || [ "${OVERWRITE}" -eq 1 ]; then
        echo "Downloading model ${MODEL_ID} to ${HF_MODEL_PATH}..."
        hf download "${MODEL_ID}" \
            --local-dir "${HF_MODEL_PATH}"
    else
        echo "Model already exists at ${HF_MODEL_PATH}, skipping download."
    fi
else
    echo "SKIP_MODEL=1, skipping model download."
fi

############################ Summary ############################

echo ""
echo "Preparation complete."
echo ""
echo "Data:"
echo "  train: ${TRAIN_FILE}"
echo "  test:  ${TEST_FILE}"
echo ""
echo "Model:"
echo "  ${HF_MODEL_PATH}"
echo ""
echo "Run training with:"
echo "  export HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "  bash ${TRAINING_SCRIPT}"
echo ""
echo "Or set RAY_DATA_HOME so the training script picks up the model automatically:"
echo "  export RAY_DATA_HOME=$(dirname "$(dirname "${HF_MODEL_PATH}")")"
echo "  bash ${TRAINING_SCRIPT}"
