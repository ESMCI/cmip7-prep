#!/bin/bash
module purge
module load ESMF/8.6.0-foss-2023a
module load Miniforge3/24.1.2-0
conda deactivate
source deactivate
conda activate /projects/NS9560K/diagnostics/cmordev_env_312/
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
