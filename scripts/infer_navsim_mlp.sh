#!/bin/bash
# OneVL MLP-head inference on NAVSIM test set.
#
# Single-process, single-GPU launcher that calls `infer_onevl.py` directly so
# logs stream straight to the terminal (good for debugging). For multi-GPU /
# multi-node sharding use `scripts/infer_navsim.sh` + `run_infer.sh`.
#
# Optional toggles (export before running):
#   DECODER_EXPLAIN=true          # also run the text aux decoder
#   VISUAL_DECODER_EXPLAIN=true   # also run the visual aux decoder
#
# Example - all three heads + aux decoders in one terminal run:
#   DECODER_EXPLAIN=true VISUAL_DECODER_EXPLAIN=true \
#       bash scripts/infer_navsim_mlp.sh
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON=${PYTHON:-python3}

# ---- Configurable (env-overrideable) ----
MODEL_PATH=${MODEL_PATH:-""}
TEST_SET_PATH=${TEST_SET_PATH:-"${ROOT_DIR}/test_data/navsim_test.json"}
IMAGE_BASE_PATH=${IMAGE_BASE_PATH:-""}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/navsim_mlp/navsim_results.json"}

DEVICE=${DEVICE:-"cuda:0"}

# ---- OneVL latent prefix (must match the trained checkpoint) ----
NUM_LATENT=${NUM_LATENT:-2}
NUM_LATENT_VIS=${NUM_LATENT_VIS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
ANSWER_PREFIX=${ANSWER_PREFIX:-"["}
PREFIX_K=${PREFIX_K:-0}

# ---- MLP float-head settings ----
FLOAT_HEAD_TYPE=${FLOAT_HEAD_TYPE:-"mlp"}
FLOAT_HEAD_OUTPUT_DIM=${FLOAT_HEAD_OUTPUT_DIM:-24}

# ---- Aux text decoder (off by default; set DECODER_EXPLAIN=true to enable) ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
AUX_VISUAL_CONDITION=${AUX_VISUAL_CONDITION:-false}
C_THOUGHT=${C_THOUGHT:-2}
MAX_EXPLAIN_TOKENS=${MAX_EXPLAIN_TOKENS:-1024}

# ---- Aux visual decoder (off by default; set VISUAL_DECODER_EXPLAIN=true) ----
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-true}
VISUAL_AUX_VISUAL_CONDITION=${VISUAL_AUX_VISUAL_CONDITION:-true}
C_THOUGHT_VISUAL=${C_THOUGHT_VISUAL:-4}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-2560}

# ---- Build extra flags ----
EXTRA_FLAGS=""
if [ -n "${FLOAT_HEAD_TYPE}" ] && [ "${FLOAT_HEAD_TYPE}" != "none" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --float_head_type ${FLOAT_HEAD_TYPE} --float_head_output_dim ${FLOAT_HEAD_OUTPUT_DIM}"
fi
if [ "${DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --decoder_explain --c_thought ${C_THOUGHT} --max_explain_tokens ${MAX_EXPLAIN_TOKENS}"
    if [ "${AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --aux_visual_condition"
    fi
fi
if [ "${VISUAL_DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --visual_decoder_explain --c_thought_visual ${C_THOUGHT_VISUAL} --max_visual_tokens ${MAX_VISUAL_TOKENS}"
    if [ "${VISUAL_AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --visual_aux_visual_condition"
    fi
fi

IMAGE_BASE_FLAG=""
if [ -n "${IMAGE_BASE_PATH}" ]; then
    IMAGE_BASE_FLAG="--image_base_path ${IMAGE_BASE_PATH}"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "=== NAVSIM MLP-Head Inference ==="
echo "  MODEL_PATH:               ${MODEL_PATH}"
echo "  TEST_SET_PATH:            ${TEST_SET_PATH}"
echo "  IMAGE_BASE_PATH:          ${IMAGE_BASE_PATH}"
echo "  OUTPUT_PATH:              ${OUTPUT_PATH}"
echo "  DEVICE:                   ${DEVICE}"
echo "  NUM_LATENT / VIS:         ${NUM_LATENT} / ${NUM_LATENT_VIS}"
echo "  ANSWER_PREFIX:            ${ANSWER_PREFIX}"
echo "  FLOAT_HEAD_TYPE:          ${FLOAT_HEAD_TYPE}"
echo "  FLOAT_HEAD_OUTPUT_DIM:    ${FLOAT_HEAD_OUTPUT_DIM}"
echo "  DECODER_EXPLAIN:          ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN:   ${VISUAL_DECODER_EXPLAIN}"
echo "  EXTRA_FLAGS:             ${EXTRA_FLAGS}"
echo "================================="

exec ${PYTHON} "${ROOT_DIR}/infer_onevl.py" \
    --model_path "${MODEL_PATH}" \
    --test_set_path "${TEST_SET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --device "${DEVICE}" \
    --num_latent ${NUM_LATENT} \
    --num_latent_vis ${NUM_LATENT_VIS} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --answer_prefix "${ANSWER_PREFIX}" \
    --prefix_k ${PREFIX_K} \
    ${IMAGE_BASE_FLAG} \
    ${EXTRA_FLAGS}
