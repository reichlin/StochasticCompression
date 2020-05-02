
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/beta*

for sigma in 0.08 0.06 0.05 0.02; do
  for decay in 0.95 0.9 0.8; do
    sbatch --export=SAMPL=1,SIGMA=$sigma,DECAY=$decay search_stable_policy.sbatch
    sleep 1
  done
done

#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

