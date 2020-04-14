
#! /bin/bash

#cd /Midgard/home/areichlin

for gamma in 1.0 0.7; do

  for depth in 3 5; do

      sbatch --export=gamma=$GAMMA,depth=$DEPTH \
             train.sbatch
      sleep 1

  done

done

