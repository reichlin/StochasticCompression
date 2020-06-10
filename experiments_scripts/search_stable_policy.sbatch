#!/usr/bin/env bash
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constrain="rivendell|khazadum"
#SBATCH --time=48:00:00

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python /Midgard/home/areichlin/compression/train.py ${SLURMD_NODENAME} ${L} ${LOSS} ${HW}


