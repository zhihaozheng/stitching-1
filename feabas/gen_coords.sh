#!/bin/bash

# Base directory
base_dir="/scratch/zhihaozheng/mec/acqs/"

# List of paths
paths=(
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-00.24.49/s2804-2023.09.27-00.24.49/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-00.05.40/s2805-2023.09.27-00.05.40/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-01.03.03/s2806-2023.09.27-01.03.03/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-01.20.58/s2807-2023.09.27-01.20.58/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-01.38.01/s2808-2023.09.27-01.38.01/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-01.55.03/s2809-2023.09.27-01.55.03/"
    "3-complete/Part1_reel1068_blade2_20230921/bladeseq-2023.09.27-02.12.10/s2810-2023.09.27-02.12.10/"
)

# Loop over each path and run the python command
for path in "${paths[@]}"; do
	python gen_stitch_coord.py "${base_dir}${path}metadata/stage_positions.csv"
done

