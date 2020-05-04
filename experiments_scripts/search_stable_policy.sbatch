#!/usr/bin/env bash
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constrain="rivendell|khazadum"
#SBATCH --time=24:00:00

source "/Midgard/home/pshi/miniconda3/etc/profile.d/conda.sh"
conda activate victor_test

python /Midgard/home/pshi/alfredo/train.py ${SLURMD_NODENAME} ${SAMPL} ${NOISE} ${EPS} ${DECAY}
