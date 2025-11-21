#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB:mpiprocs=128
#PBS -N fullLmon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe

NCPUS=$(cat $PBS_NODEFILE | wc -l)
echo "Running on $NCPUS"

module load conda
conda activate CMORDEV
poetry run python ./scripts/monthly_cmor.py --realm lnd --test --workers $NCPUS "$@"
