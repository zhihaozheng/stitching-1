#!/bin/bash
#SBATCH -t 9:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mem=1000GB

module load cudatoolkit/12.2
module load anaconda3/2023.3 
conda activate stiching

export PYTHONUNBUFFERED=TRUE

python -u stitch.py Part1_reel1068_blade1_20230727/bladeseq-2023.08.01-19.14.39/s1257-2023.08.01-19.14.39 --output_dir runs/reel1068_blade1_s1257 --no_upload
