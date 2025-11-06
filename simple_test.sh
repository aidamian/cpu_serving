#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
fi

HF_TOKEN_VALUE="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "error: set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in the environment or .env" >&2
  exit 1
fi

export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_VALUE}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
if [[ -z "${VIRTUALENV_HOME:-}" ]]; then
  if [[ -d "/opt/venvs" ]]; then
    export VIRTUALENV_HOME="/opt/venvs"
  else
    export VIRTUALENV_HOME="${ROOT_DIR}/.venvs"
  fi
else
  export VIRTUALENV_HOME="${VIRTUALENV_HOME}"
fi
export UV_CACHE_DIR="${ROOT_DIR}/.uv_cache"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export HOME="${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${ROOT_DIR}/src"

RUN_LABEL="${RUN_LABEL:-simple-test}"
OUTPUT_DIR="${ROOT_DIR}/artifacts/simple_test"
PROMPT_TEXT="${PROMPT:-Write the DDL SQL for the definition of user accounts table. Output only the viable SQL.}"
MAX_NEW_TOKENS_VALUE="${MAX_NEW_TOKENS:-250}"
HF_NUM_THREADS_VALUE="${HF_NUM_THREADS:-2}"
VLLM_NUM_THREADS_VALUE="${VLLM_NUM_THREADS:-2}"
LLAMACPP_NUM_THREADS_VALUE="${LLAMACPP_NUM_THREADS:-2}"
DEFAULT_HF_MODEL_PATH="${ROOT_DIR}/models/meta-llama--Llama-3.2-1B"
DEFAULT_HF_REPO_ID="meta-llama/Llama-3.2-1B"
HF_REPO_ID="${HF_REPO_ID:-${DEFAULT_HF_REPO_ID}}"
MODEL_ID="${MODEL_ID:-${HF_MODEL_ID:-${DEFAULT_HF_MODEL_PATH}}}"
DEFAULT_LLAMACPP_DIR="${ROOT_DIR}/models/hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
LLAMACPP_MODEL_PATH="${LLAMACPP_MODEL_PATH:-${DEFAULT_LLAMACPP_DIR}/llama-3.2-1b-instruct-q4_k_m.gguf}"
LLAMACPP_REPO_ID="${LLAMACPP_REPO_ID:-hugging-quants/Llama-3.2-1B-Instruct-GGUF}"
DEFAULT_LLAMACPP_Q8_DIR="${ROOT_DIR}/models/hugging-quants--Llama-3.2-1B-Instruct-Q8_0-GGUF"
DEFAULT_LLAMACPP_Q8_PATH="${DEFAULT_LLAMACPP_Q8_DIR}/llama-3.2-1b-instruct-q8_0.gguf"
LLAMACPP_Q8_MODEL_PATH="${LLAMACPP_Q8_MODEL_PATH:-${DEFAULT_LLAMACPP_Q8_PATH}}"
SYNC_VENVS="${SIMPLE_TEST_SYNC_VENVS:-0}"
CONTINUE_ON_ERROR="${SIMPLE_TEST_CONTINUE_ON_ERROR:-1}"

ORIGINAL_MODEL_ID="${MODEL_ID}"
if [[ "${MODEL_ID}" == */* && "${MODEL_ID}" != "${ROOT_DIR}/models/"* && "${MODEL_ID}" != /* ]]; then
  if [[ "${HF_REPO_ID}" == "${DEFAULT_HF_REPO_ID}" ]]; then
    HF_REPO_ID="${ORIGINAL_MODEL_ID}"
  fi
  MODEL_ID="${ROOT_DIR}/models/${MODEL_ID//\//--}"
fi
if [[ -z "${VLLM_MODEL_ID:-}" || "${VLLM_MODEL_ID}" == "${ORIGINAL_MODEL_ID}" ]]; then
  VLLM_MODEL_ID="${MODEL_ID}"
fi

_venv_python_path() {
  local venv_dir="$1"
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "${venv_dir}/Scripts/python.exe"
  else
    echo "${venv_dir}/bin/python"
  fi
}

ensure_venv_ready() {
  local name="$1"
  local dir="${VIRTUALENV_HOME}/venv-${name}"
  local python_bin
  python_bin="$(_venv_python_path "${dir}")"
  if [[ ! -x "${python_bin}" ]]; then
    cat <<EOF >&2
error: expected virtualenv for backend '${name}' at ${dir}
       Run 'python scripts/setup_virtualenvs.py ${name}' to create it.
EOF
    exit 1
  fi
}

run_download_model() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2

  local python_bin
  python_bin="$(_venv_python_path "${VIRTUALENV_HOME}/venv-hf")"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="python3"
  fi

  local extra_args=()
  while (($#)); do
    extra_args+=("$1")
    shift
  done

  set +e
  "${python_bin}" "${ROOT_DIR}/scripts/download_model.py" \
    --model-id "${repo_id}" \
    --local-dir "${local_dir}" \
    "${extra_args[@]}"
  local status=$?
  set -e
  return "${status}"
}

ensure_hf_model() {
  if [[ -d "${MODEL_ID}" ]]; then
    return
  fi

  if [[ "${MODEL_ID}" != "${ROOT_DIR}/models/"* ]]; then
    return
  fi

  echo "Downloading Hugging Face model snapshot '${HF_REPO_ID}'..." >&2
  run_download_model "${HF_REPO_ID}" "${MODEL_ID}" || true
}

ensure_llamacpp_model() {
  if [[ -f "${LLAMACPP_MODEL_PATH}" ]]; then
    return
  fi

  if [[ "${LLAMACPP_MODEL_PATH}" != "${ROOT_DIR}/models/"* ]]; then
    return
  fi

  local target_dir
  target_dir="$(dirname "${LLAMACPP_MODEL_PATH}")"
  local target_file
  target_file="$(basename "${LLAMACPP_MODEL_PATH}")"

  echo "Downloading primary llama.cpp GGUF '${target_file}'..." >&2
  run_download_model "${LLAMACPP_REPO_ID}" "${target_dir}" --allow-pattern "${target_file}" || true
  if [[ -f "${LLAMACPP_MODEL_PATH}" ]]; then
    return
  fi

  local hf_cli="${VIRTUALENV_HOME}/venv-hf/bin/huggingface-cli"
  if [[ -x "${hf_cli}" ]]; then
    set +e
    "${hf_cli}" download "${LLAMACPP_REPO_ID}" \
      --include "${target_file}" \
      --revision main \
      --local-dir "${target_dir}" \
      --local-dir-use-symlinks False >/dev/null 2>&1
    set -e
  fi
}

ensure_venv_ready "hf"
ensure_venv_ready "vllm"
ensure_venv_ready "llamacpp"

ensure_hf_model

if [[ ! -d "${MODEL_ID}" ]]; then
  cat <<EOF >&2
error: expected Hugging Face model directory at ${MODEL_ID}
       Provide MODEL_ID or HF_MODEL_ID pointing to a local repository or enable network access.
EOF
  exit 1
fi

ensure_llamacpp_model

if [[ ! -f "${LLAMACPP_MODEL_PATH}" ]]; then
  echo "error: expected llama.cpp model at ${LLAMACPP_MODEL_PATH}" >&2
  echo "Set LLAMACPP_MODEL_PATH to a valid GGUF file for the simple test." >&2
  exit 1
fi

ensure_llamacpp_q8_model() {
  if [[ -f "${LLAMACPP_Q8_MODEL_PATH}" ]]; then
    return
  fi

  local discovered
  discovered="$(find "${ROOT_DIR}/models" -maxdepth 3 -type f -name '*q8*.gguf' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${discovered}" ]]; then
    LLAMACPP_Q8_MODEL_PATH="${discovered}"
    export LLAMACPP_Q8_MODEL_PATH
    return
  fi

  mkdir -p "${DEFAULT_LLAMACPP_Q8_DIR}"
  local q8_filename
  q8_filename="$(basename "${DEFAULT_LLAMACPP_Q8_PATH}")"
  if [[ "${LLAMACPP_Q8_MODEL_PATH}" == "${DEFAULT_LLAMACPP_Q8_PATH}" ]]; then
    echo "Downloading llama.cpp int8 GGUF '${q8_filename}'..." >&2
    run_download_model "${LLAMACPP_REPO_ID}" "${DEFAULT_LLAMACPP_Q8_DIR}" --allow-pattern "${q8_filename}" || true
    if [[ -f "${DEFAULT_LLAMACPP_Q8_PATH}" ]]; then
      LLAMACPP_Q8_MODEL_PATH="${DEFAULT_LLAMACPP_Q8_PATH}"
      export LLAMACPP_Q8_MODEL_PATH
      return
    fi
  fi
  local hf_cli="${VIRTUALENV_HOME}/venv-hf/bin/huggingface-cli"
  local repos=(
    "hugging-quants/Llama-3.2-1B-Instruct-GGUF"
    "TheBloke/Llama-3.2-1B-Instruct-GGUF"
  )

  if [[ -x "${hf_cli}" ]]; then
    for repo in "${repos[@]}"; do
      set +e
      "${hf_cli}" download "${repo}" \
        --include "llama-3.2-1b-instruct-q8_0.gguf" \
        --revision main \
        --local-dir "${DEFAULT_LLAMACPP_Q8_DIR}" \
        --local-dir-use-symlinks False >/dev/null 2>&1
      local status=$?
      set -e
      if (( status == 0 )); then
        break
      fi
    done
  fi

  if [[ -f "${DEFAULT_LLAMACPP_Q8_PATH}" ]]; then
    LLAMACPP_Q8_MODEL_PATH="${DEFAULT_LLAMACPP_Q8_PATH}"
    export LLAMACPP_Q8_MODEL_PATH
    return
  fi

  cat <<EOF >&2
error: unable to locate an int8 GGUF model for llama.cpp.
       Expected file at ${LLAMACPP_Q8_MODEL_PATH} or discoverable under ${ROOT_DIR}/models.
       Ensure you have permission to download llama-3.2-1b-instruct-q8_0.gguf and place it locally.
EOF
  exit 1
}

run_unit_tests() {
  echo "Running quantization unit tests..." >&2
  PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m unittest tests.test_quantized_benchmarks
}

ensure_llamacpp_q8_model
run_unit_tests

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CLI_ARGS=("$@")

has_flag() {
  local flag="$1"
  for arg in "${CLI_ARGS[@]}"; do
    case "${arg}" in
      "${flag}"|"${flag}"=*) return 0 ;;
    esac
  done
  return 1
}

add_default_option() {
  local flag="$1"
  local value="$2"
  if ! has_flag "${flag}"; then
    EXTRA_ARGS+=("${flag}" "${value}")
  fi
}

add_default_flag() {
  local flag="$1"
  if ! has_flag "${flag}"; then
    EXTRA_ARGS+=("${flag}")
  fi
}

EXTRA_ARGS=()

add_default_option "--label" "${RUN_LABEL}"
add_default_option "--output-dir" "${OUTPUT_DIR}"

if ! has_flag "--prompt" && ! has_flag "--prompt-file"; then
  add_default_option "--prompt" "${PROMPT_TEXT}"
fi

add_default_option "--max-new-tokens" "${MAX_NEW_TOKENS_VALUE}"
add_default_option "--hf-model-id" "${MODEL_ID}"
add_default_option "--hf-num-threads" "${HF_NUM_THREADS_VALUE}"
if [[ -d "${MODEL_ID}" ]]; then
  add_default_option "--hf-tokenizer-id" "${MODEL_ID}"
fi
add_default_option "--vllm-model-id" "${VLLM_MODEL_ID}"
add_default_option "--vllm-num-threads" "${VLLM_NUM_THREADS_VALUE}"
add_default_flag "--vllm-enforce-eager"
add_default_option "--llamacpp-model-path" "${LLAMACPP_MODEL_PATH}"
add_default_option "--llamacpp-num-threads" "${LLAMACPP_NUM_THREADS_VALUE}"
if [[ -f "${LLAMACPP_Q8_MODEL_PATH}" ]]; then
  if ! has_flag "--llamacpp-quantization"; then
    EXTRA_ARGS+=("--llamacpp-quantization" "int8=${LLAMACPP_Q8_MODEL_PATH}")
  fi
  if ! has_flag "--llamacpp-quantization-order"; then
    EXTRA_ARGS+=("--llamacpp-quantization-order" "int4" "int8")
  fi
  if ! has_flag "--llamacpp-quantization-name"; then
    EXTRA_ARGS+=("--llamacpp-quantization-name" "int4")
  fi
fi
add_default_flag "--print-json"

if [[ "${SYNC_VENVS}" != "1" ]]; then
  add_default_flag "--skip-venv-sync"
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  add_default_flag "--continue-on-error"
fi

set +e
python3 "${ROOT_DIR}/scripts/run_all_benchmarks.py" \
  "${EXTRA_ARGS[@]}" \
  "${CLI_ARGS[@]}"
status=$?
set -e

if (( status != 0 )); then
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    echo "warning: simple test completed with partial failures (exit code ${status}). See logs for details." >&2
  else
    exit "${status}"
  fi
fi
