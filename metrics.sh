#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --job-name=CalculatingMetrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%A.out

cd /home/scur2445/ai4mi_project

source ai4mi/bin/activate

python metrics.py