import click
from pathlib import Path
import subprocess
from datetime import datetime

@click.command()
@click.argument('start', type=int)
@click.argument('interval', type=int)
@click.argument('end', type=int)
@click.option('--reel', default=1068, help='Reel number to filter by')
@click.option('--top-dir', default='/scratch/zhihaozheng/mec/acqs', help='Top directory to search.')
@click.option('--tif-path', default='/scratch/rmorey/mec/tiff_sections/part1_reel1068_blade2', help='Target path to store TIFF files in.')
@click.option('--parallel', default=60, help='Number of parallel threads for temu.')
@click.option('--lst-file-path', default='sections.lst', help='Output .lst file name.')
@click.option('--sh-file-path', default='convert_to_tif.sh', help='Output shell script file name.')
@click.option('--push', default=True, help='Send push notifications')
def find_sections(start, interval, end, reel, top_dir, tif_path, parallel, lst_file_path, sh_file_path, push):

    sequence = list(range(start, end + 1, interval))

    if Path(lst_file_path).exists() or Path(sh_file_path).exists():
        raise FileExistsError(f"{lst_file_path} or {sh_file_path} already exists.")
    
    with open(lst_file_path, 'w') as lst_file, open(sh_file_path, 'w') as sh_file:
        
        sh_file.write("#!/bin/bash\n")
        sh_file.write("#SBATCH --job-name=tif\n")
        sh_file.write("#SBATCH --output=./slurm-%N.%j.out\n")
        sh_file.write("#SBATCH --error=./slurm-%N.%j.err\n")
        sh_file.write("#SBATCH --nodes=1\n")
        sh_file.write("#SBATCH --ntasks=1\n")
        sh_file.write("#SBATCH --exclusive\n")
        sh_file.write("#SBATCH --mem=0\n")
        sh_file.write("#SBATCH --time=99:00:00\n")
        sh_file.write("#SBATCH --mail-type=all\n")
        sh_file.write("#SBATCH --mail-user=rmorey@princeton.edu\n")
        sh_file.write(f"export TIFF_PATH={tif_path}\n")
        
        for seq_num in sequence:
            cmd = ['fd', '--exact-depth', '4', '-t', 'd', str(seq_num), top_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            stdout_cleaned = result.stdout.strip()

            if not stdout_cleaned:
                raise ValueError(f"No match found for {seq_num}")

            # Filter results by reel number
            filtered_results = [line for line in stdout_cleaned.split('\n') if f'reel{reel}' in line]
            if not filtered_results:
                continue

            if len(filtered_results) > 1:
                filtered_results.sort(key=get_timestamp_from_path, reverse=True)
            
            latest_match = filtered_results[0]
            last_dir = Path(latest_match).name
            sh_file.write(f'temu --parallel {parallel} totif {latest_match} $TIFF_PATH/{last_dir}\n')
            if push:
                sh_file.write(f'~/push converted {last_dir}\n')
            lst_file.write(f'{last_dir}\n')

def get_timestamp_from_path(path):
    try:
        timestamp_str = Path(path).name.split('s')[-1]
        timestamp = datetime.strptime(timestamp_str, '%Y.%m.%d-%H.%M.%S')
    except (IndexError, ValueError):
        return datetime.min
    return timestamp

if __name__ == '__main__':
    find_sections()
