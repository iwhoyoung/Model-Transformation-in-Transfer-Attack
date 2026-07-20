#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SECONDS=0

# Override these from the shell when needed, for example:
# MODELS_STR="resnet18 resnet50" GPU_ID=1 BATCHSIZE=4 bash run_transform_sweep.sh
ATTACK="${ATTACK:-simattack}"
MODELS_STR="${MODELS_STR:-resnet18 densenet121 inception_v3 vit_base_patch16_224 vit_small_patch16_224}"
TRANSFORM_NUMS_STR="${TRANSFORM_NUMS_STR:-1 5 10 20 50 100 200 500 1000 2000}"
INPUT_DIR="${INPUT_DIR:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./adv_data}"
EPOCH="${EPOCH:-10}"
EPS="${EPS:-16}"
ALPHA="${ALPHA:-1.6}"
BATCHSIZE="${BATCHSIZE:-8}"
EVAL_BATCHSIZE="${EVAL_BATCHSIZE:-32}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

read -r -a MODELS <<< "${MODELS_STR}"
read -r -a TRANSFORM_NUMS <<< "${TRANSFORM_NUMS_STR}"

LOG_DIR="${OUTPUT_ROOT}/logs"
SUMMARY_FILE="${OUTPUT_ROOT}/runtime_summary.csv"

mkdir -p "${LOG_DIR}"
printf "model,transform_num,train_seconds,eval_seconds,output_dir\n" > "${SUMMARY_FILE}"

format_duration() {
  local total_seconds="$1"
  printf "%dh %dm %ds" "$((total_seconds / 3600))" "$((total_seconds % 3600 / 60))" "$((total_seconds % 60))"
}

for model in "${MODELS[@]}"; do
  echo "############################################################"
  echo "Proxy model: ${model}"
  echo "############################################################"

  for transform_num in "${TRANSFORM_NUMS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${model}_${ATTACK}_tn${transform_num}_epoch${EPOCH}_eps${EPS}_alpha${ALPHA}/${model}"
    log_prefix="${LOG_DIR}/${model}_tn${transform_num}"

    echo "============================================================"
    echo "Train: attack=${ATTACK}, model=${model}, transform_num=${transform_num}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    train_start=${SECONDS}
    python main.py \
      --attack "${ATTACK}" \
      --model "${model}" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${output_dir}" \
      --batchsize "${BATCHSIZE}" \
      --epoch "${EPOCH}" \
      --transform_num "${transform_num}" \
      --eps "${EPS}" \
      --alpha "${ALPHA}" \
      --GPU_ID "${GPU_ID}" \
      --seed "${SEED}" \
      --num_workers "${NUM_WORKERS}" \
      2>&1 | tee "${log_prefix}_train.log"
    train_seconds=$((SECONDS - train_start))

    echo "Train time for model=${model}, transform_num=${transform_num}: $(format_duration "${train_seconds}")"

    echo "============================================================"
    echo "Eval: attack=${ATTACK}, model=${model}, transform_num=${transform_num}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    eval_start=${SECONDS}
    python main.py \
      --eval \
      --attack "${ATTACK}" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${output_dir}" \
      --batchsize "${EVAL_BATCHSIZE}" \
      --GPU_ID "${GPU_ID}" \
      --seed "${SEED}" \
      --num_workers "${NUM_WORKERS}" \
      2>&1 | tee "${log_prefix}_eval.log"
    eval_seconds=$((SECONDS - eval_start))

    echo "Eval time for model=${model}, transform_num=${transform_num}: $(format_duration "${eval_seconds}")"
    printf "%s,%s,%s,%s,%s\n" "${model}" "${transform_num}" "${train_seconds}" "${eval_seconds}" "${output_dir}" >> "${SUMMARY_FILE}"
  done
done

duration=${SECONDS}
echo "=========================================="
echo "All runs finished."
echo "Runtime summary: ${SUMMARY_FILE}"
echo "Total time: $(format_duration "${duration}")"
echo "Total seconds: ${duration}"
echo "=========================================="
