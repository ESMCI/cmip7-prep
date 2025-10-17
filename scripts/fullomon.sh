#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -N fullamon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe


module load conda
conda activate CMORDEV

poetry run python ./scripts/monthly_cmor.py --realm ocn --test --workers 1 --cmip-vars sos --skip-timeseries

#/glade/u/home/cmip7/cases/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1 /glade/work/hannay/cesm_tags/cesm3_0_beta06/cime
