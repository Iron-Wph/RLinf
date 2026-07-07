#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/debug_robotwin_cameras.py"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export HYDRA_FULL_ERROR=1

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/mnt/public2/wph/codes/develop_async/RoboTwin"}
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-"/mnt/public2/wph/models/robotwin_assets"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

CONFIG_NAME=${1:-"robotwin_camera_debug_franka_single_adjusted"}
OUTPUT_DIR=${2:-"${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"}

mkdir -p "${OUTPUT_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} debug.output_dir=${OUTPUT_DIR}"
echo "${CMD}"
${CMD} 2>&1 | tee "${OUTPUT_DIR}/camera_debug.log"

