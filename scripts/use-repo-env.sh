#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source scripts/use-repo-env.sh" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_venv_path="$repo_root/.venv312"
if [[ ! -e "$default_venv_path" && -e "$repo_root/.venv" ]]; then
  default_venv_path="$repo_root/.venv"
fi
venv_path="${LIBERO_VENV_PATH_OVERRIDE:-$default_venv_path}"

export LIBERO_CONFIG_PATH="$repo_root/.libero"
export LIBERO_VENV_PATH="$venv_path"
export LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-$repo_root/datasets}"
export XDG_CACHE_HOME="$repo_root/.cache/xdg"
export XDG_DATA_HOME="$repo_root/.cache/xdg-data"
export TMPDIR="$repo_root/.cache/tmp"
export PIP_CACHE_DIR="$repo_root/.cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
export UV_CACHE_DIR="$repo_root/.cache/uv"
export HF_HOME="$repo_root/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$repo_root/.cache/torch"

mkdir -p \
  "$LIBERO_CONFIG_PATH" \
  "$LIBERO_DATASET_PATH" \
  "$XDG_CACHE_HOME" \
  "$XDG_DATA_HOME" \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME"

if command -v findmnt >/dev/null 2>&1; then
  mount_options="$(findmnt -no OPTIONS -T "$venv_path" 2>/dev/null || true)"
  if [[ "$mount_options" == *noexec* ]]; then
    echo "warning: $venv_path is on a noexec mount; compiled Python packages will not import from that location" >&2
  fi
fi

if [[ -d "$venv_path" && "${VIRTUAL_ENV:-}" != "$venv_path" ]]; then
  # shellcheck disable=SC1091
  source "$venv_path/bin/activate"
fi

if [[ "${VIRTUAL_ENV:-}" == "$venv_path" ]]; then
  python -m pip config --site set global.cache-dir "$PIP_CACHE_DIR" >/dev/null
fi

echo "LIBERO repo environment loaded from $repo_root"
echo "venv path: $venv_path"
echo "pip cache: $PIP_CACHE_DIR"
echo "huggingface cache: $HF_HOME"