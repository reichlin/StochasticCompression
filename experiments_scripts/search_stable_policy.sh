
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_1*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_3*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_0_epsdecay_0.5
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_0_epsdecay_0.8

for eps in 0.5 0.2 0.05 0.01; do
  for decay in 0.25 0.4 0.64; do
    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
    sleep 1
  done
done

