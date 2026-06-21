#!/usr/bin/env bash
# Build TissueForge from source INTO the active conda/pixi env prefix (linux-64).
#
# Invoked by the pixi task `build-tf` (linux-64 target override) from the default
# environment:
#     pixi run build-tf
#
# This is the linux-64 port of build_tissue_forge_osx.sh. Per
# docs/MIGRATION_to_windows_wsl2.md §4, the macOS-only CMake bits are DROPPED:
#   - -DCMAKE_APPLE_SILICON_PROCESSOR (Apple-silicon arch force)
#   - the CMAKE_OSX_DEPLOYMENT_TARGET / CMAKE_OSX_SYSROOT blocks
#   - -DTF_ENABLE_AVX:BOOL=OFF / -DTF_ENABLE_SSE4:BOOL=OFF  (were forced off for
#     Apple silicon; on x86_64 we leave TF's defaults ON -- CMakeLists.txt:78-79 --
#     for SIMD speed, which matters for the long sweeps).
# Everything else is kept verbatim: prefix/find-root/install all = $CONDA_PREFIX,
# Python_EXECUTABLE, LIBXML_INCLUDE_DIR, Ninja, --target install.
#
# Why build into $CONDA_PREFIX: tissue-forge's CMake computes its python install
# dir relative to the chosen prefix (tissue-forge/CMakeLists.txt:302-312), so with
# CMAKE_INSTALL_PREFIX=$CONDA_PREFIX it lands in this env's site-packages and links
# against the SAME assimp/glfw/eigen/libxml2 the env already provides -- no second
# conda env, no rpath bridging between envs.
set -euo pipefail

# --- locate things -----------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
TFSRCDIR="${TFSRCDIR:-${SCRIPT_DIR}/tissue-forge}"          # source tree (this workspace)
TFBUILDDIR="${TFBUILDDIR:-${SCRIPT_DIR}/tissue-forge_build}" # out-of-tree build dir
TFBUILD_CONFIG="${TFBUILD_CONFIG:-Release}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX is unset. Run this through pixi: 'pixi run build-tf'." >&2
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
# linux-64 (x86_64, gcc). No Apple-silicon arch force, no OSX deployment/sysroot,
# and AVX/SSE4 left at TF's defaults (ON) for SIMD speed.
declare -a ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE:STRING="${TFBUILD_CONFIG}"
  -DCMAKE_PREFIX_PATH:PATH="${CONDA_PREFIX}"
  -DCMAKE_FIND_ROOT_PATH:PATH="${CONDA_PREFIX}"
  -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}"
  -DPython_EXECUTABLE:PATH="${CONDA_PREFIX}/bin/python"
  -DLIBXML_INCLUDE_DIR:PATH="${CONDA_PREFIX}/include/libxml2"
)

cmake "${ARGS[@]}" "${TFSRCDIR}"

# --- build + install ----------------------------------------------------------
cmake --build . --config "${TFBUILD_CONFIG}" --target install

echo
echo "*TF* Build + install complete. Validate with:  pixi run verify"
