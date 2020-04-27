
#! /bin/bash

rm -r /Midgard/home/pshi/alfredo/experiments_scripts/slurm*
rm -r /Midgard/home/pshi/alfredo/experiments_scripts/policy_log/eps*

sbatch --export=BETA=0.1,SIZE=32,DEPTH=6,JOIN=1,ACT=1 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=0.1,SIZE=32,DEPTH=6,JOIN=0,ACT=0 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=0.1,SIZE=32,DEPTH=8,JOIN=1,ACT=0 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=0.1,SIZE=64,DEPTH=6,JOIN=1,ACT=0 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=0.1,SIZE=64,DEPTH=8,JOIN=1,ACT=0 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=0.01,SIZE=32,DEPTH=6,JOIN=1,ACT=0 search_stable_policy.sbatch
sleep 1
sbatch --export=BETA=10.0,SIZE=32,DEPTH=6,JOIN=1,ACT=0 search_stable_policy.sbatch
sleep 1

#for eps in 0.5 0.2 0.05 0.01; do
#  for decay in 0.25 0.4 0.64; do
#    sbatch --export=EPS=$eps,EPSDECAY=$decay search_stable_policy.sbatch
#    sleep 1
#  done
#done

