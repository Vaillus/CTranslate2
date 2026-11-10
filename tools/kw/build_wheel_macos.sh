#!/bin/bash
# Build the macOS arm64 CPU wheel of the fork (Accelerate + ruy, no OpenMP),
# for Python 3.12, into wheelhouse/. Run from the repo root on an Apple
# Silicon Mac with cmake and uv installed. Mirrors upstream's
# python/tools/prepare_build_environment_macos.sh for arm64, installing the
# C++ library under install/ instead of /usr/local.
#
# Python build tools: upstream's python/install_requirements.txt
# (pybind11==2.11.1, the version upstream builds and tests its binding
# with) plus delocate==0.13.0. CMake is not pinned: the one on the machine
# is used (4.2.1 when this script was written), with
# CMAKE_POLICY_VERSION_MINIMUM=3.5 as upstream sets it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

git submodule update --init --depth 1 third_party/cxxopts third_party/cpu_features \
    third_party/spdlog third_party/ruy third_party/googletest
git -C third_party/ruy submodule update --init --depth 1 third_party/cpuinfo

mkdir -p build-release
cmake -S . -B build-release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI=OFF \
    -DCMAKE_INSTALL_PREFIX="$ROOT/install" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_RUY=ON
cmake --build build-release -j "$(sysctl -n hw.ncpu)"
cmake --install build-release

uv venv -q --allow-existing -p 3.12 .venv-build
uv pip install -q -p .venv-build/bin/python -r python/install_requirements.txt \
    delocate==0.13.0

cp README.md python/
rm -rf python/build python/dist
(cd python && CTRANSLATE2_ROOT="$ROOT/install" MACOSX_DEPLOYMENT_TARGET=11.0 \
    ../.venv-build/bin/python setup.py -q bdist_wheel)
DYLD_LIBRARY_PATH="$ROOT/install/lib" .venv-build/bin/delocate-wheel \
    -w wheelhouse python/dist/ctranslate2-*.whl
ls wheelhouse
