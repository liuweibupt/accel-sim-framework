#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUTLASS_DIR="${REPO_ROOT}/modal/extern/cutlass"
CUTLASS_REPO_URL="https://github.com/NVIDIA/cutlass.git"
PINNED_REVISION="f7b19de32c5d1f3cedfc735c2849f12b537522ee" # v3.5.1

mkdir -p "$(dirname "${CUTLASS_DIR}")"

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  echo "[fetch_cutlass] Cloning CUTLASS into ${CUTLASS_DIR}"
  git clone --filter=blob:none "${CUTLASS_REPO_URL}" "${CUTLASS_DIR}"
else
  echo "[fetch_cutlass] CUTLASS checkout already exists at ${CUTLASS_DIR}"
fi

echo "[fetch_cutlass] Checking out pinned revision ${PINNED_REVISION}"
git -C "${CUTLASS_DIR}" fetch --force --tags origin
git -C "${CUTLASS_DIR}" checkout --force "${PINNED_REVISION}"

echo "[fetch_cutlass] Ready at revision $(git -C "${CUTLASS_DIR}" rev-parse HEAD)"
