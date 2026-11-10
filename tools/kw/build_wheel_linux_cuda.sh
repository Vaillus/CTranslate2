#!/bin/bash
# Build the Linux x86_64 CUDA 12 wheel of the fork, for Python 3.12, into
# wheelhouse/. Run from the repo root on a Linux x86_64 host with uv and a
# container engine (podman or docker; set CIBW_CONTAINER_ENGINE, default
# podman, which runs rootless).
#
# Mirrors upstream's CI job build-python-wheels for Linux: cibuildwheel
# 3.2.1 in the manylinux_2_28 image, with upstream's
# python/tools/prepare_build_environment_linux.sh (CUDA 12.8, cuDNN off,
# oneDNN, MKL, OpenMPI, CUDA_ARCH_LIST=Common) as its before-all step, and
# auditwheel's default repair, which packs libctranslate2 into the wheel.
#
# One change to upstream's script: the CUDA architectures. With CUDA 12.8
# and the CMake 3.22 upstream installs, "Common" ends at 8.6+PTX: no
# SASS for Blackwell. CMake 3.22's
# cuda_select_nvcc_arch_flags only parses one-digit major versions, so 12.0
# cannot go in CUDA_ARCH_LIST; it goes in CUDA_NVCC_FLAGS as a -gencode for
# sm_120 (needs CUDA >= 12.8). The script writes that variant of upstream's
# file to python/tools/prepare_build_environment_linux_kw.sh (generated,
# untracked) and leaves upstream's file untouched.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

git submodule update --init --recursive --depth 1

SM120='-gencode=arch=compute_120,code=sm_120'
src=python/tools/prepare_build_environment_linux.sh
dst=python/tools/prepare_build_environment_linux_kw.sh
grep -q -- '-DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all"' "$src"
sed "s#-DCUDA_NVCC_FLAGS=\"-Xfatbin=-compress-all\"#-DCUDA_NVCC_FLAGS=\"-Xfatbin=-compress-all;$SM120\"#" \
    "$src" > "$dst"
chmod +x "$dst"

export CIBW_CONTAINER_ENGINE="${CIBW_CONTAINER_ENGINE:-podman}"
export CIBW_BUILD="cp312-manylinux_x86_64"
export CIBW_ARCHS=x86_64
export CIBW_ENVIRONMENT_PASS_LINUX=CIBW_ARCHS
export CIBW_BEFORE_ALL_LINUX="$dst"
export CIBW_BEFORE_BUILD="pip install -r python/install_requirements.txt"
export CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28
export CIBW_SKIP="*-musllinux_*"

uv tool run cibuildwheel==3.2.1 --platform linux --output-dir wheelhouse python
