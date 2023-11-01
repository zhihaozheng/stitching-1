#!/bin/bash
#SBATCH -t 9:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mem=1000GB

module load cudatoolkit/12.2
module load anaconda3/2023.3 
conda activate stiching

export PYTHONUNBUFFERED=TRUE

# Make sure the user passed two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_image_dir> <output_dir>"
    exit 1
fi

python -u stitch.py $1 --output_dir $2 --no_upload
