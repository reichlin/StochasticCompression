
#! /bin/bash

#rm /Midgard/home/areichlin/compression/slurm*

#rm -r /Midgard/home/areichlin/compression/pareto_experiments/Pareto_default_experiment_*
#rm -r /Midgard/home/areichlin/compression/pareto_experiments/delete

for threshold in 0.9 0.95 0.99; do
  for gamma in 0.1 0.01; do
    sbatch --export=THRESHOLD=$threshold,GAMMA=$gamma experiments_scripts/transformer_testing.sbatch
    sleep 1
  done
done