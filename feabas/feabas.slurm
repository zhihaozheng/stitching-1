#!/usr/bin/env bash
#SBATCH -J 'feabas'
#SBATCH -o %x-%j.out
#SBATCH --time=9-00:00:00
#SBATCH -w sarekl16-5
#SBATCH --exclusive
#SBATCH --mem=0


source /etc/bashrc
module load anacondapy/2023.03
source ~/feabas-venv/bin/activate
python ~/src/feabas/scripts/stitch_main.py --mode matching 

python ~/src/feabas/scripts/stitch_main.py --mode optimization

python ~/src/feabas/scripts/stitch_main.py --mode rendering

