#!/bin/bash

#SBATCH --job-name=MoCo
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=Logs/moco_%j_%x.out
#SBATCH --error=Logs/moco_%j_%x.err
#SBATCH --qos=long
#SBATCH --time=48:00:00 

module load miniconda

source /app/miniconda/24.1.2/etc/profile.d/conda.sh

conda activate CFP

python3 train.py
