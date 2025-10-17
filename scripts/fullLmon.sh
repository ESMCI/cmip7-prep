#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -N fullLmon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe
module purge
module load intel cray-mpich esmf/8.9.0 netcdf-mpi
module load conda
if [ -z "$CONDA_PREFIX" ]; then
    module load conda
    conda activate CMORDEV2
fi
DIR_TO_ADD="$CONDA_PREFIX/lib/python3.13/site-packages/"
if [[ ":$PYTHONPATH:" != *":$DIR_TO_ADD:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DIR_TO_ADD"
fi
poetry run python ./scripts/monthly_cmor.py --realm lnd --test --workers 1 --skip-timeseries --cmip-vars lai

#/glade/u/home/cmip7/cases/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1 /glade/work/hannay/cesm_tags/cesm3_0_beta06/cime
