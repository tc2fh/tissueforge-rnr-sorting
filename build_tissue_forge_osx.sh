#!/usr/bin/env bash
# Build TissueForge from source INTO the active conda/pixi env prefix (osx-arm64).
#
# Invoked by the pixi task `build-tf` from the `tfsource` environment:
#     pixi run -e tfsource build-tf
#
# Why build into $CONDA_PREFIX: tissue-forge's CMake computes its python install
# dir relative to the chosen prefix (tissue-forge/CMakeLists.txt:302-312), so with
# CMAKE_INSTALL_PREFIX=$CONDA_PREFIX it lands in this env's site-packages and links
# against the SAME assimp/glfw/eigen/libxml2 the env already provides -- no second
# conda env, no DYLD/@rpath bridging between envs.
#
# Mirrors package/local/osx/install_core.sh, but repointed at the pixi env instead
# of the project's standalone tissue-forge_install/env.
set -euo pipefail

# --- locate things -----------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
TFSRCDIR="${TFSRCDIR:-${SCRIPT_DIR}/tissue-forge}"          # source tree (this workspace)
TFBUILDDIR="${TFBUILDDIR:-${SCRIPT_DIR}/tissue-forge_build}" # out-of-tree build dir
TFBUILD_CONFIG="${TFBUILD_CONFIG:-Release}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX is unset. Run this through pixi: 'pixi run -e tfsource build-tf'." >&2
  exit 1
fi
if [[ ! -f "${TFSRCDIR}/CMakeLists.txt" ]]; then
  echo "ERROR: no CMakeLists.txt at TFSRCDIR=${TFSRCDIR}" >&2
  exit 2
fi

echo "*TF* source : ${TFSRCDIR}"
echo "*TF* build  : ${TFBUILDDIR}"
echo "*TF* install: ${CONDA_PREFIX}  (the active pixi env)"
echo "*TF* python : $(python -c 'import sys; print(sys.version.split()[0])')"

mkdir -p "${TFBUILDDIR}"
cd "${TFBUILDDIR}"

# --- configure ----------------------------------------------------------------
# Apple-silicon flags copied from package/local/osx/install_core.sh:43-49
# (force arm64, disable AVX/SSE4 -- TF defaults them ON in CMakeLists.txt:78-79).
declare -a ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE:STRING="${TFBUILD_CONFIG}"
  -DCMAKE_PREFIX_PATH:PATH="${CONDA_PREFIX}"
  -DCMAKE_FIND_ROOT_PATH:PATH="${CONDA_PREFIX}"
  -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}"
  -DPython_EXECUTABLE:PATH="${CONDA_PREFIX}/bin/python"
  -DLIBXML_INCLUDE_DIR:PATH="${CONDA_PREFIX}/include/libxml2"
  -DCMAKE_APPLE_SILICON_PROCESSOR:STRING=arm64
  -DTF_ENABLE_AVX:BOOL=OFF
  -DTF_ENABLE_SSE4:BOOL=OFF
)

# Honor the conda compilers' deployment target if the env activation set one.
if [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
  ARGS+=(-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="${MACOSX_DEPLOYMENT_TARGET}")
fi
# Honor the conda SDK sysroot if present (CONDA_BUILD_SYSROOT / SDKROOT).
_sysroot="${CONDA_BUILD_SYSROOT:-${SDKROOT:-}}"
if [[ -n "${_sysroot}" && -d "${_sysroot}" ]]; then
  ARGS+=(-DCMAKE_OSX_SYSROOT:PATH="${_sysroot}")
fi

cmake "${ARGS[@]}" "${TFSRCDIR}"

# --- build + install ----------------------------------------------------------
cmake --build . --config "${TFBUILD_CONFIG}" --target install

echo
echo "*TF* Build + install complete. Validate with:  pixi run -e tfsource verify"
