#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/workspace/venv/bin/python3"
query_script="$script_dir/query-offload.py"

answers_dir="answers"
params_path="params.cfg"
adapter_dir=""
start_checkpoint=""
forward_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --answers-dir)
            answers_dir="$2"
            shift 2
            ;;
        --params-path)
            params_path="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --adapter-dir)
            adapter_dir="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --start-checkpoint)
            start_checkpoint="$2"
            shift 2
            ;;
        *)
            forward_args+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                forward_args+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
    esac
done

resolve_param() {
    local key="$1"
    local file="$2"
    awk -F= -v key="$key" '
        $1 == key {
            value = $2
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            gsub(/^"/, "", value)
            gsub(/"$/, "", value)
            print value
            exit
        }
    ' "$file"
}

if [[ ! -e "$params_path" ]]; then
    if [[ -e "$script_dir/$params_path" ]]; then
        params_path="$script_dir/$params_path"
    fi
fi

if [[ -z "$adapter_dir" ]]; then
    adapter_dir="$(resolve_param "ft_output_dir" "$params_path")"
fi

if [[ -z "$adapter_dir" ]]; then
    echo "ERROR: impossible de determiner adapter_dir" >&2
    exit 1
fi

if [[ -e "$answers_dir" ]]; then
    echo "ERROR: le repertoire existe deja: $answers_dir"
    exit 1
fi

mkdir -p "$answers_dir"

mapfile -t checkpoint_dirs < <(
    find "$adapter_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
)

models=("base")
for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    models+=("$(basename "$checkpoint_dir")")
done
models+=("final")

if [[ -n "$start_checkpoint" ]]; then
    start_checkpoint="$(basename "$start_checkpoint")"
    sliced=()
    found=0
    for model_name in "${models[@]}"; do
        if [[ "$model_name" == "$start_checkpoint" ]]; then
            found=1
        fi
        if [[ $found -eq 1 ]]; then
            sliced+=("$model_name")
        fi
    done
    if [[ ${#sliced[@]} -eq 0 ]]; then
        echo "ERROR: checkpoint introuvable: $start_checkpoint"
        exit 1
    fi
    models=("${sliced[@]}")
fi

for model_name in "${models[@]}"; do
    staging_dir="$answers_dir/.staging-$model_name"
    rm -rf "$staging_dir"

    if [[ "$model_name" == "base" ]]; then
        start_arg="base"
        end_arg="base"
    else
        start_arg="$model_name"
        end_arg="$model_name"
    fi

    "$python_bin" "$query_script" \
        --answers-dir "$staging_dir" \
        --params-path "$params_path" \
        --adapter-dir "$adapter_dir" \
        --start-checkpoint "$start_arg" \
        --end-checkpoint "$end_arg" \
        "${forward_args[@]}"

    shopt -s nullglob
    files=("$staging_dir"/*)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "ERROR: aucun fichier genere pour $model_name"
        exit 1
    fi
    mv "$staging_dir"/* "$answers_dir"/
    rmdir "$staging_dir"
done
