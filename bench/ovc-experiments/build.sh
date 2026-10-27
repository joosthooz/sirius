#!/usr/bin/env bash
# Builds the offset-value-code experiments against the libcudf in the pixi environment.
# Usage: pixi run bench/ovc-experiments/build.sh [extra nvcc args...]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="${CONDA_PREFIX:?run through pixi run}"
arch="${OVC_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"
out_dir="${OVC_OUT_DIR:-${here}/../../build/ovc-experiments}"
mkdir -p "${out_dir}"
case "$(uname -m)" in
  aarch64) host=aarch64 cuda_target=sbsa-linux ;;
  *) host=x86_64 cuda_target=x86_64-linux ;;
esac

nvcc -std=c++20 -O3 --extended-lambda --expt-relaxed-constexpr \
  -gencode "arch=compute_${arch},code=sm_${arch}" \
  -ccbin "${env_prefix}/bin/${host}-conda-linux-gnu-c++" \
  -I "${env_prefix}/include/rapids" -I "${env_prefix}/include" \
  -I "${env_prefix}/targets/${cuda_target}/include" \
  "$@" \
  "${here}/ovc_bench.cu" \
  -L "${env_prefix}/lib" -L "${env_prefix}/targets/${cuda_target}/lib" \
  -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker "${env_prefix}/lib" \
  -lcudf -lrmm -lcudart -ldl -lpthread \
  -o "${out_dir}/ovc_bench"
echo "built ${out_dir}/ovc_bench"
