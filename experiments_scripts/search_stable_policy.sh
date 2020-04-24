
#! /bin/bash

for decay in 0.8 0.5 0.4; do
  sbatch --export=SAMPL=0,EPSDECAY=$decay search_stable_policy.sbatch
  sleep 1
done

sbatch --export=SAMPL=1,EPSDECAY=0.0 search_stable_policy.sbatch
sleep 1

for decay in 0.9 0.5 0.1; do
  sbatch --export=SAMPL=3,EPSDECAY=$decay search_stable_policy.sbatch
  sleep 1
done

