#!/bin/bash
# Must be sourced so the environment persists in the calling shell
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Usage: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi
module purge
module load ESMF/8.6.0-foss-2023a
module load Miniforge3/24.1.2-0
# Load conda's shell functions for this shell only (no `conda init` / .bashrc edits)
eval "$(conda shell.bash hook)"
conda deactivate
conda activate /projects/NS9560K/diagnostics/cmordev_env_312/
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
