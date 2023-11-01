#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 48

# Make sure the user passed argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_image_dir>"
    echo "<input_image_dir> should be the subtiles directory for the section"
    exit 1
fi

parallel -j 48 convert -compress LZW {} tiff/{.}.tiff ::: $1/*.bmp