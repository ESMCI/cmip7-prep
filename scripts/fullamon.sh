#!/usr/bin/env bash
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -N fullamon_cmor_processing
#PBS -A CESM0024
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -j oe

module load conda
conda activate /glade/work/cmip7/conda-envs/CMOR

python /glade/work/cmip7/cmip7-prep/scripts/atm_monthly.py /glade/u/home/cmip7/cases/b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1 /glade/work/hannay/cesm_tags/cesm3_0_beta06/cime
