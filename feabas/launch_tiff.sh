#!/bin/bash

for script in tiff_scripts/*.sh; do
    sbatch "$script"
done

