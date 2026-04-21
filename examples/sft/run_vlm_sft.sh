#! /bin/bash

export VLM_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$VLM_PATH"))
export SRC_FILE="${VLM_PATH}/train_vlm_sft.py"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

CONFIG_NAME="qwen3_subtask_sft_vlm"

IDX=$1   # 传入 i
IDX_PADDED=$(printf "%04d" ${IDX})
echo "IDX_PADDED: ${IDX_PADDED}"
LOG_DIR="${REPO_PATH}/logs/EVAL-all-tasks-${IDX}-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_vlm_sft.log"

mkdir -p "${LOG_DIR}"

EVAL_PATH="/mnt/project_rlinf_hs/wph/datasets/behaviour/task-${IDX_PADDED}/subtask_sft_eval.parquet"

CMD="python ${SRC_FILE} \
  --config-path ${VLM_PATH}/config/ \
  --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${LOG_DIR} \
  data.eval_data_paths=${EVAL_PATH}"

echo ${CMD} > ${MEGA_LOG_FILE}
eval ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}