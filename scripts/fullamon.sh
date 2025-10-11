#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -N fullamon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe

module load  ncarenv/24.12  gcc/12.4.0  cray-mpich/8.1.29 esmf/8.9.0
module load conda
conda activate CMORDEV2
DIR_TO_ADD="$CONDA_PREFIX/lib/python3.13/site-packages/"
if [[ ":$PYTHONPATH:" != *":$DIR_TO_ADD:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DIR_TO_ADD"
fi
echo $PYTHONPATH
poetry run python ./scripts/monthly_cmor.py --realm atm --test --workers 128 --overwrite

#/glade/u/home/cmip7/cases/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1 /glade/work/hannay/cesm_tags/cesm3_0_beta06/cime
