#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=400GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

module load cudatoolkit/12.2
module load anaconda3/2023.3 
conda activate stiching

export PYTHONUNBUFFERED=TRUE

python -u stitch.py Part1_reel1068_blade1_20230727/bladeseq-2023.08.01-19.14.39/s1257-2023.08.01-19.14.39 --output_dir runs/test_run_large --no_render
