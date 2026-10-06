#!/usr/bin/env bash
# Build the AP Novo primary environment and smoke-test it on one GPU.
#
# Target: a workstation with internet access and an NVIDIA GPU (driver >= 525
# for the CUDA 12.9 wheels this pulls in). Run this on a machine that can reach
# PyPI and GitHub -- compute nodes on an offline cluster cannot, so build the
# venv on a login node or a dev box and move it, or point PIP at a local mirror.
#
# Usage:
#   bash tev_cold/setup_env.sh [install_dir]
#
# Deliberately NOT doing: the LigandMPNN environment (separate Python 3.11 venv,
# see README "LigandMPNN Environment"), and the stock af3.bin.zst weights. Add
# those only if you need them.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${1:-$REPO_DIR}"
VENV="$INSTALL_DIR/.venv"

log() { printf '\n=== %s\n' "$*"; }

log "Preflight"
command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
else
  echo "WARNING: nvidia-smi not found. The install will still work but you will" >&2
  echo "         be running on CPU, which is far too slow for a real campaign." >&2
fi

log "Creating Python 3.12 venv at $VENV"
# 3.12 specifically: pyproject requires >=3.12, and the pinned rdkit/tokamax
# wheels are not published for every newer interpreter.
uv venv --python 3.12 "$VENV"

log "Installing AP Novo (pulls jax 0.10.2 + CUDA 12 wheels + AlphaFold 3)"
# Installing the package itself brings in the pinned jax rather than the looser
# ">=0.4.30" the top-level README suggests. Do not pre-install jax by hand.
VIRTUAL_ENV="$VENV" uv pip install -e "$REPO_DIR"

log "Compiling AlphaFold 3 chemical component data"
# build_data writes ccd.pickle and chemical_component_sets.pickle, without which
# AF3 cannot parse ligands. If components.cif is not discoverable, LIBCIFPP_DATA_DIR
# must point at the directory holding it.
if ! "$VENV/bin/build_data"; then
  echo "build_data failed. Locate components.cif and retry, e.g.:" >&2
  echo "  find / -name components.cif 2>/dev/null" >&2
  echo "  LIBCIFPP_DATA_DIR=/path/to/dir $VENV/bin/build_data" >&2
  exit 1
fi

log "Downloading model weights (~1.5 GB total)"
# AP Novo generator weights are non-commercial use only; AF3-LA weights carry the
# AlphaFold 3 parameters terms. See WEIGHTS_TERMS_OF_USE.md before redistributing
# -- you may not share either set outside your organization.
mkdir -p "$INSTALL_DIR/models/apnovo_generator" "$INSTALL_DIR/models/af3_la"
[ -f "$INSTALL_DIR/models/apnovo_generator/generator.bin.zst" ] || \
  wget -P "$INSTALL_DIR/models/apnovo_generator" \
    https://storage.googleapis.com/alphaprotein_novo/generator.bin.zst
[ -f "$INSTALL_DIR/models/af3_la/af3_leaving_atom.bin.zst" ] || \
  wget -P "$INSTALL_DIR/models/af3_la" \
    https://storage.googleapis.com/alphafold3/af3_leaving_atom.bin.zst

log "Verifying JAX sees the GPU"
"$VENV/bin/python" - <<'PY'
import jax
print('jax', jax.__version__)
devs = jax.devices()
print('devices:', devs)
if not any(d.platform == 'gpu' for d in devs):
    print('WARNING: no GPU visible to JAX -- check driver/CUDA before running a campaign')
PY

log "Smoke test: reduced-step Kemp eliminase campaign on one GPU"
# 2 designs x 50 sampling steps. This exercises generation -> folding ->
# evaluation but skips LigandMPNN, which needs the second environment.
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$VENV/bin/python" run_pipeline.py \
    --manifest=examples/kemp_eliminase/kemp_test_manifest.json \
    --output_dir="$INSTALL_DIR/smoke_test_kemp" \
    --apn_model_dir="$INSTALL_DIR/models/apnovo_generator" \
    --af3_model_dir="$INSTALL_DIR/models/af3_la" \
    --only_stage=generation

log "Done"
cat <<EOF
Generation-stage smoke test output: $INSTALL_DIR/smoke_test_kemp

If AF3 folding later fails with a Triton/attention kernel error on a consumer
GPU (RTX 4090 is compute capability 8.9, while AF3's Triton kernels target 8.0
and 9.0), the model config is built in run_alphafold.py:256 -- set the
flash-attention implementation on config.global_config to the XLA backend there.
On an A800 (8.0) this should not arise.
EOF
