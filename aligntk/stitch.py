import os
import subprocess
import sys

from temu import convert_to_tif, getpairsbatch, getscriptbatch


# Set variables
USER_DIR = "/mnt/sink/scratch/rmorey/mec"
VOXA_DIR = "/home/voxa/scripts"  # not actually voxa
REEL = "reel1073"
PARALLEL = 20

os.environ["USER_DIR"] = USER_DIR
os.environ["VOXA_DIR"] = VOXA_DIR
os.environ["REEL"] = REEL

# Function to run shell commands
def run_command(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"Error: {' '.join(cmd)}: {err.strip()}")
    return out.strip()


# 2. Generate the file path patterns
#TODO configure as a variable
os.chdir(f"{USER_DIR}/stitching_test")

for i in range(10, 31):
    dirs_to_create = [
        f"cmaps/s0{i}",
        f"maps/s0{i}",
        f"amaps/s0{i}",
        f"grids/s0{i}",
        f"aligned/s0{i}",
        f"imaps/s0{i}",
    ]

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

# 2. Create logs dirs required by alignTK:
i = "10-30"
os.makedirs(f"{USER_DIR}/sbatch_scripts/{REEL}/stitching_test", exist_ok=True)
os.chdir(f"{USER_DIR}/sbatch_scripts/{REEL}/stitching_test")
logs_subdirs = ["rst_reg/logs", "align/logs", "hres/logs", "imap/logs"]

for d in logs_subdirs:
    os.makedirs(f"s0{i}/{d}", exist_ok=True)

#  convert images to .tiff 
convert_to_tif(
    f"{USER_DIR}/acqs/bladeseq-2023.03.19-15.16.23/s020-2023.03.19-15.16.23",
    f"{USER_DIR}/tif/{REEL}",
    PARALLEL
)

#  take a list of sections (e.g. i) and generate pairs.lst for alignTK (generates /scratch/zhihaozheng/mec/stitching_test/lst/s..pairs.lst) --stage_step, a supplied parameter different for different microscopes. Usually stable for months. Could get from stage_position.csv. Right now (x_step + y+setp)/2

getpairsbatch(
    acqs="VOXA_DIR/mec_stitching/stitching_test/acq_20.lst",
    mpath=USER_DIR+"/stitching_test/lst",
    tile_path=USER_DIR+"/tif/"+REEL,
    pos_path=USER_DIR+"/stage_positions/"+REEL+"_blade2",
    stage_step=55243
)


getscriptbatch(
    img=VOXA_DIR+"/mec_stitching/stitching_test/acq_20.lst",
    tif_path=USER_DIR+"/tif/"+REEL,
    output=USER_DIR+"/sbatch_scripts/"+REEL+"/stitching_test/rst_reg/s020",
    sbatch=VOXA_DIR+"/stitch/stitching/220409_stitch_full_section/sbatch_rst_reg_template.sh",
    rst=True,
    register=True,
    align=False,
    imap=False,
    apply_map_red=False,
    apply_map_hres=False,
    mpi=56,
    map_path=USER_DIR+"/stitching_test"
)

exit()

# TODO: sbatch those scripts to sarek

# 9. Generate sarek .sh scripts for `align`
cmd_9 = f"""
temu getscriptbatch --img {VOXA_DIR}/mec_stitching/stitching_test/acq_20.lst \
--align \
--output {USER_DIR}/sbatch_scripts/{REEL}/stitching_test/align/s020 \
--sbatch {VOXA_DIR}/stitch/stitching/220409_stitch_full_section/sbatch_align_template.sh \
--tif_path {USER_DIR}/tif/{REEL} \
--map_path {USER_DIR}/stitching_test
"""
run_command(cmd_9)

# 10. Again, running sarek scripts may need direct invocation or shell call.

# ... Additional steps ...

# 13. Update the stitched tiles to neuroglancer
run_command("pip install git+https://github.com/seung-lab/tem2ng.git@stitch_upload")
cmd_13 = f"tem2ng -p 24 upload s020/imap/ tigerdata://sseung-archive/mec/prelim_stitching/ --z 20 --pad 122880"
run_command(cmd_13)

# 14. Downsample with corgie
cmd_14 = f"""
corgie downsample --src_layer_spec '{
{
"type": "img",
"path": "tigerdata://sseung-archive/mec/prelim_stitching"
}
}' \
--mip_start 0 --mip_end 7 --mips_per_task 4 \
--start_coord "0, 0, 20" --end_coord "655360,655360,21" \
--chunk_xy 2048 --verbose --queue_name zhihao-seamless
"""
run_command(cmd_14)
