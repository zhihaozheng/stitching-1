#!/usr/bin/env bash

# Array of section indicies
sections=(0 1 2 3 4 5 6 7)

# Loop through each section
for i in "${sections[@]}"; do
    # Submit a separate SLURM job for each section
    sbatch <<EOT
#!/usr/bin/env bash
#SBATCH -J "feabas-$i"
#SBATCH -o "feabas-$i-%j.out"
#SBATCH --time=9-00:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=64


source /etc/bashrc
module load anacondapy/2023.03
source ~/feabas-venv/bin/activate

# Run all three steps for the specific section

python ~/src/feabas/scripts/stitch_main.py --mode matching --start $i --stop $(($i + 1)) --step 1

python ~/src/feabas/scripts/stitch_main.py --mode optimization --start $i --stop $(($i + 1)) --step 1 

python ~/src/feabas/scripts/stitch_main.py --mode rendering --start $i --stop $(($i + 1)) --step 1

EOT
done
