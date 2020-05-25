
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_1_dump_pi_prob

for epochs in 1 5; do
  sbatch --export=EPOCHS=epochs,KSAMPL=1,CSAMPL=0,ADAPT=0 search_stable_policy.sbatch
  sleep 1
  sbatch --export=EPOCHS=epochs,KSAMPL=1,CSAMPL=1,ADAPT=0 search_stable_policy.sbatch
  sleep 1
  sbatch --export=EPOCHS=epochs,KSAMPL=1,CSAMPL=2,ADAPT=0 search_stable_policy.sbatch
  sleep 1
  sbatch --export=EPOCHS=epochs,KSAMPL=1,CSAMPL=1,ADAPT=1 search_stable_policy.sbatch
  sleep 1
  sbatch --export=EPOCHS=epochs,KSAMPL=1,CSAMPL=2,ADAPT=1 search_stable_policy.sbatch
  sleep 1
done


#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

