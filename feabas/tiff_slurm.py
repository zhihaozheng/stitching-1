EMAIL = "rmorey@princeton.edu"
TIFF_PATH = "/scratch/rmorey/mec/tiff_sections/reel1068_blade2"

def generate_individual_slurm_scripts(input_file, script_folder):
    with open(input_file, 'r') as file:
        paths = file.readlines()


    for path in paths:
        path = path.strip()
        section = path.split('/')[-2]


        script_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={section}\n"
            f"#SBATCH --output=./slurm-%N.%j.out\n"
            f"#SBATCH --error=./slurm-%N.%j.err\n"
            f'#SBATCH --cpus-per-task=16\n'
            f"#SBATCH --exclusive\n"
            f"#SBATCH --mem=0\n"
            f"#SBATCH --time=99:00:00\n"
            f"#SBATCH --mail-type=all\n"
            f"#SBATCH --mail-user={EMAIL}\n"
            f"export TIFF_PATH={TIFF_PATH}\n\n"
            f"source /usr/people/rm5876/temu-venv/bin/activate\n"
            f"temu --parallel 16 totif /scratch/zhihaozheng/mec/acqs/{path} $TIFF_PATH/{section}\n"
            f"~/push converted {section}\n"
        )

        script_filename = f"{script_folder}/slurm_script_{section}.sh"
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)

generate_individual_slurm_scripts("sections.txt", "tiff_scripts")
