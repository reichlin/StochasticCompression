
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_1_dump_pi_prob

for alpha in 0.5 1.16 2.0; do
  for delta in 0.1 0.2 0.5 0.8; do
    sbatch --export=ALPHA=$alpha,DELTA=$delta search_stable_policy.sbatch
    sleep 1
  done
done


#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

