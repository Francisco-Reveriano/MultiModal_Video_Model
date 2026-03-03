#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/output/nohup_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/video_fine_tuning_${TIMESTAMP}.txt"
PID_FILE="${LOG_DIR}/video_fine_tuning_${TIMESTAMP}.pid"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
TARGET_SCRIPT="${PROJECT_ROOT}/src/video_fine_tuning.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter not found or not executable at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Error: target script not found at ${TARGET_SCRIPT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
nohup "${PYTHON_BIN}" "${TARGET_SCRIPT}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started video fine-tuning in background."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo
echo "Tail logs with:"
echo "  tail -f \"${LOG_FILE}\""
