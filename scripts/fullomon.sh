#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -N fullOmon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe


module load conda
conda activate CMORDEV

poetry run python ./scripts/monthly_cmor.py --realm ocn --test --workers 1 --cmip-vars sos --skip-timeseries
