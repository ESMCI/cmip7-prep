
# cmip7-prep

Quickstart:
1) module load conda
2) on derecho (cesm):
   > conda activate /glade/work/jedwards/conda-envs/CMORDEV
   on nird (noresm):
   > coonda activate /projects/NS9560K/diagnostics/cmordev_env/
3) make sure that you have generated timeseries files for the run
4) on derecho (cesm):
   > qcmd -- bash scripts/fullamon.sh
