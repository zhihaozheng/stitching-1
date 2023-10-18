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

# Make sure the user passed two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_image_dir> <output_dir>"
    exit 1
fi

python -u $1 --output_dir $2 --no_render
