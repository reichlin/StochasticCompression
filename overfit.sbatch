#!/usr/bin/env bash
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constrain="rivendell|khazadum"
#SBATCH --time=48:00:00

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python /Midgard/home/areichlin/compression/overfit.py ${SLURMD_NODENAME}

conda deactivate