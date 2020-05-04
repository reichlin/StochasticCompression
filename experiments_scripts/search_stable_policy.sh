
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/sampling_policy_1_dump_pi_prob

sbatch --export=SAMPL=4,NOISE=0,EPS=0.0,DECAY=0.0 search_stable_policy.sbatch
sleep 1

sbatch --export=SAMPL=4,NOISE=1,EPS=0.1,DECAY=0.5 search_stable_policy.sbatch
sleep 1

sbatch --export=SAMPL=4,NOISE=1,EPS=0.3,DECAY=0.1 search_stable_policy.sbatch
sleep 1

sbatch --export=SAMPL=4,NOISE=2,EPS=0.1,DECAY=0.5 search_stable_policy.sbatch
sleep 1

sbatch --export=SAMPL=4,NOISE=2,EPS=0.05,DECAY=0.5 search_stable_policy.sbatch
sleep 1

#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

