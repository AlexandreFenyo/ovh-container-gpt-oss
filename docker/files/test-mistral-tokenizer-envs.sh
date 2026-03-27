#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.tmp-mistral-tokenizer-envs}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Ministral-8B-Instruct-2410}"

CONFIGS=(
  "4.55.0|0.21.4"
  "4.55.0|0.22.2"
  "5.3.0|0.22.2"
)

echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "WORK_DIR=$WORK_DIR"
echo "TOKENIZER_NAME=$TOKENIZER_NAME"
echo

mkdir -p "$WORK_DIR"

for config in "${CONFIGS[@]}"; do
  IFS="|" read -r transformers_version tokenizers_version <<< "$config"
  env_name="tf-${transformers_version}_tok-${tokenizers_version}"
  env_dir="$WORK_DIR/$env_name"

  echo "============================================================"
  echo "TESTING transformers=$transformers_version tokenizers=$tokenizers_version"
  echo "VENV=$env_dir"

  rm -rf "$env_dir"
  "$PYTHON_BIN" -m venv "$env_dir"

  "$env_dir/bin/pip" install --upgrade pip >/dev/null
  if ! "$env_dir/bin/pip" install \
    "transformers==$transformers_version" \
    "tokenizers==$tokenizers_version" \
    sentencepiece >/dev/null; then
    echo "INSTALL_RESULT: error"
    echo
    continue
  fi

  echo
  "$env_dir/bin/python" "$ROOT_DIR/docker/files/diagnose-mistral-tokenizer.py" \
    --tokenizer-name "$TOKENIZER_NAME" || true
  echo
done

echo "DONE"
