#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_COMMIT=050dde50c9d70cf207db84f7224eedc491d817b2
LLAMA_CPP_DIR="${DEPLOYMENT_LLAMA_CPP_DIR:-/data/experiment/LQ/llama.cpp-deployment-gguf-v1}"
CONVERT_VENV="${DEPLOYMENT_LLAMA_CPP_CONVERT_VENV:-${LLAMA_CPP_DIR}-convert-venv}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

if [[ -n "$(git -C "$LLAMA_CPP_DIR" status --porcelain)" ]]; then
  echo "Refusing to alter a dirty llama.cpp checkout: $LLAMA_CPP_DIR" >&2
  exit 1
fi

git -C "$LLAMA_CPP_DIR" fetch origin "$LLAMA_CPP_COMMIT"
git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_COMMIT"

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$(nproc)"

if [[ ! -x "$CONVERT_VENV/bin/python" ]]; then
  python3 -m venv "$CONVERT_VENV"
fi
"$CONVERT_VENV/bin/python" -m pip install --upgrade pip
"$CONVERT_VENV/bin/python" -m pip install \
  -r "$LLAMA_CPP_DIR/requirements.txt"

echo "LLAMA_CPP_DIR=$LLAMA_CPP_DIR"
echo "LLAMA_CPP_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "DEPLOYMENT_LLAMA_CPP_PYTHON=$CONVERT_VENV/bin/python"
