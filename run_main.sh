#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=ModelTraining
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output_%A.out

cd /home/scur2445/ai4mi_project

source ai4mi/bin/activate

#python main.py --dataset SEGTHOR --mode full --epoch 25 --num_workers 8 --dest results/segthor/enet --gpu
python main.py --dataset SEGTHOR --mode full --test --num_workers 8 --dest results/segthor/enet --gpu

#python main.py --dataset SEGTHOR --mode full --epoch 25 --num_workers 8 --dest results/segthor/unet --gpu
