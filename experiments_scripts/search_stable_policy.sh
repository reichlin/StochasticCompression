
#! /bin/bash

#rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
#rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_1_dump_pi_prob

rm /Midgard/home/areichlin/compression/slurm*

rm -r /Midgard/home/areichlin/compression/pareto_experiments/Pareto_default_experiment_*
rm -r /Midgard/home/areichlin/compression/pareto_experiments/delete

for L in 2 4 6 8 12; do
  sbatch --export=L=$L,LOSS=0,HW=168 experiments_scripts/search_stable_policy.sbatch
  sleep 1
done

sbatch --export=L=8,LOSS=1,HW=168 experiments_scripts/search_stable_policy.sbatch
sleep 1
sbatch --export=L=8,LOSS=0,HW=200 experiments_scripts/search_stable_policy.sbatch
sleep 1


#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

