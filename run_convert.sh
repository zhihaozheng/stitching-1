#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 48

WORK_DIR=/scratch/gpfs/dmturner/stitching/Part1_reel1068_blade1_20230727/bladeseq-2023.08.01-19.14.39/s1257-2023.08.01-19.14.39/subtiles

parallel -j 48 convert -compress LZW {} tiff/{.}.tiff ::: ${WORK_DIR}/*.bmp