Code for stitching sections using [SOFIMA](https://github.com/google-research/sofima/)

## setup

```
conda env create -f environment.yml 
conda activate stitching
python -m pip install -r requirements.txt
```
If that doesn't work, try:

```
conda env create -n stitching
conda activate stitching
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install numpy pandas pillow cloud-volume igneous-pipeline task-queue git+https://github.com/davidt0x/sofima.git
```

## usage
```
python stitch.py /scratch/zhihaozheng/mec/acqs/3-complete/Part1_reel1068_blade1_20230727/bladeseq-2023.08.01-19.14.39/s1257-2023.08.01-19.14.39
```

It will log time elapsed for each of the major operations, and save the stitched section to `reel{reel}_blade{blade}_s{section}_{datestamp}_stitched.npy`
