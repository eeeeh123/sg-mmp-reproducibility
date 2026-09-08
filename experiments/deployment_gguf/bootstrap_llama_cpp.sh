#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_COMMIT=050dde50c9d70cf207db84f7224eedc491d817b2
LLAMA_CPP_DIR="${DEPLOYMENT_LLAMA_CPP_DIR:-/data/experiment/LQ/llama.cpp-deployment-gguf-v1}"
CONVERT_VENV="${DEPLOYMENT_LLAMA_CPP_CONVERT_VENV:-${LLAMA_CPP_DIR}-convert-venv}"
BUILD_JOBS="${DEPLOYMENT_BUILD_JOBS:-4}"

for tool in git cmake python3 "${CC:-cc}" "${CXX:-c++}" "${CUDACXX:-nvcc}"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required build tool: $tool" >&2
    exit 1
  fi
done

if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEPLOYMENT_BUILD_JOBS must be a positive integer, got: $BUILD_JOBS" >&2
  exit 1
fi

if ! "${CXX:-c++}" -std=c++17 -mavx2 -x c++ -fsyntax-only - >/dev/null 2>&1 <<'CPP'; then
#include <charconv>
#include <immintrin.h>
int main() {
    char buffer[16];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), 1);
    const __m128 half = _mm_setzero_ps();
    const __m256 whole = _mm256_set_m128(half, half);
    return result.ec == std::errc{} && _mm256_movemask_ps(whole) == 0 ? 0 : 1;
}
CPP
  echo "The selected C++ compiler/libstdc++ cannot build pinned llama.cpp." >&2
  echo "Use a modern host compiler (tested target: GCC/G++ 12) via CC, CXX, and CUDAHOSTCXX." >&2
  exit 1
fi

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
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$BUILD_JOBS"

if [[ ! -x "$CONVERT_VENV/bin/python" ]]; then
  python3 -m venv "$CONVERT_VENV"
fi
"$CONVERT_VENV/bin/python" -m pip install --upgrade pip
"$CONVERT_VENV/bin/python" -m pip install \
  -r "$LLAMA_CPP_DIR/requirements.txt"

echo "LLAMA_CPP_DIR=$LLAMA_CPP_DIR"
echo "LLAMA_CPP_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "DEPLOYMENT_LLAMA_CPP_PYTHON=$CONVERT_VENV/bin/python"
