#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TrainingTest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

cd /home/scur2445/ai4mi_project

source ai4mi/bin/activate

python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu