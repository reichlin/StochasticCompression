#!/usr/bin/env bash
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constrain="rivendell|khazadum"
#SBATCH --time=48:00:00

source "/Midgard/home/pshi/miniconda3/etc/profile.d/conda.sh"
conda activate victor_test

python /Midgard/home/pshi/alfredo/train.py ${SLURMD_NODENAME} ${EPOCHS} ${KSAMPL} ${CSAMPL} ${ADAPT}
