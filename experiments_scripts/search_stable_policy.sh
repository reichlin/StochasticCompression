
#! /bin/bash

for adv in 0 1; do

  for ksize in 1 3; do

    for eps in 0.0 0.1; do

      for clip in 0 1; do

          sbatch --export=ADV=adv,KSIZE=ksize,EPS=eps,CLIP=clip \
                 train.sbatch
          sleep 1

      done

    done

  done

done

