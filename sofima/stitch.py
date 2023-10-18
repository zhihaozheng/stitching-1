import argparse
import logging
from datetime import datetime
from time import time
import re
import os
import pathlib
import pickle

import logging

from typing import Mapping

import click
import functools as ft
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from sofima import stitch_elastic, stitch_rigid, mesh, flow_utils, warp
import pandas as pd
from cloudvolume import CloudVolume
from igneous.task_creation import create_downsampling_tasks
from taskqueue import LocalTaskQueue
from functools import partial, wraps


from typing import Callable, Any

CLOUDVOLUME_PATH = "matrix://bucket-test"

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def usable_cpu_count():
    """Get number of CPUs usable by the current process.

    Takes into consideration cpusets restrictions.

    Returns
    -------
    int
    """
    try:
        result = len(os.sched_getaffinity(0))
    except AttributeError:
        try:
            result = len(psutil.Process().cpu_affinity())
        except AttributeError:
            result = os.cpu_count()
    return result


def save_pickle(obj: Any, filename: str):
    """Save an object to a binary pickle using highest supported protocol"""
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str):
    """Load an object from a binary pickle"""
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)

    return obj


def run_with_cache(cache_dir: pathlib.Path, func: Callable, *args, **kwargs):
    """
    Simple function that runs a function and caches the result in a directory. 
    If the result is already in the cache directory, it loads it instead.
    """

    cache_file = cache_dir / f"{func.__name__}.pkl"

    # IF the results aren't already cached, run the function and cache results
    if not cache_file.exists():
        try:
            result = func(*args, **kwargs)
            save_pickle(obj=result, filename=str(cache_file))
        except Exception as ex:
            logger.error(f"Error running {func.__name__}")
            raise ex
    else:
        logger.info(f'Skipping {func.__name__}, loading form {str(cache_file)}')
        result = load_pickle(filename=str(cache_file))

    return result


def with_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f'{func.__name__} took {elapsed_time:.0f} seconds to execute.')
        return result
    return wrapper

def generate_supertile_map_for_section(section_path) -> np.ndarray:
    """
    Generate a 2D array of supertile_ids from the stage_positions.csv file
    """

    logger.info("Generate super tile map for section.")

    csv_path = f"{section_path}/metadata/stage_positions.csv"

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Correct the column names by removing leading spaces
    df.columns = [col.strip() for col in df.columns]

    # Normalize the stage_x_nm and stage_y_nm values
    df["norm_stage_x"] = df["stage_x_nm"].rank(method="dense").astype(int) - 1
    df["norm_stage_y"] = df["stage_y_nm"].rank(method="dense").astype(int) - 1

    # Determine the dimensions of the 2D array
    max_x = df["norm_stage_x"].max()
    max_y = df["norm_stage_y"].max()

    # Initialize an array of shape (max_y+1, max_x+1) with None values
    arr = np.full((max_y + 1, max_x + 1), None)

    # Populate the array with tile_id values using numpy's advanced indexing
    arr[df["norm_stage_y"].values, df["norm_stage_x"].values] = df["tile_id"].values

    # Reverse the order of the rows in the array
    arr_reversed = arr[::-1]

    return arr_reversed

def generate_tile_id_map(supertile_map) -> np.ndarray:
    """
    Generate a 2D array of tile_ids from the supertile_map
    """

    # Cricket subtile order
    SUBTILE_MAP = [[6, 7, 8], [5, 0, 1], [4, 3, 2]]

    tile_id_map = []
    for supertile_row in supertile_map:
        for subtile_row in SUBTILE_MAP:
            current_row = []
            for supertile in supertile_row:
                for subtile in subtile_row:
                    if supertile is None:
                        current_row.append(None)
                    else:
                        current_row.append(f"{supertile:04}_{subtile}")
            tile_id_map.append(current_row)
    return np.array(tile_id_map)

@with_timer
def load_tiles(tile_id_map: np.ndarray, path_to_section: str) -> Mapping[tuple[int, int], np.ndarray]:
    """Load the tiles from disk and return a map of tile_id -> tile_image"""
    logger.info("Loading tiles from disk")
    tile_map = {}

    for y in range(tile_id_map.shape[0]):
        for x in range(tile_id_map.shape[1]):
            tile_id = tile_id_map[y, x]
            if tile_id is None:
                continue
            # logger.info(f"Loading {tile_id}")
            with open(f"{path_to_section}/subtiles/tile_{tile_id}.bmp", "rb") as fp:
                img = Image.open(fp)
                tile = np.array(img)
                # if the tile is uniform, skip it
                if np.all(tile == tile[0,0]):
                    continue
                tile_map[(x, y)] = tile

    if len(tile_map) == 0:
        logger.error("Only black tiles found! Can't stich empty set of tiles.")
        raise ValueError("Only black tiles found! Can't stich empty set of tiles.")

    return tile_map

@with_timer
def compute_coarse_tile_positions(
        tile_space, 
        tile_map):
    
    min_overlap_x = 700
    max_overlap_x = 1200
    min_overlap_y = 700
    max_overlap_y = 1200

    overlaps_xy = (tuple(range(min_overlap_x, max_overlap_x, 100)), tuple(range(min_overlap_y, max_overlap_y, 100)))

    logger.info("Computing coarse tile positions")

    coarse_offsets_x, coarse_offsets_y = stitch_rigid.compute_coarse_offsets(
        tile_space, tile_map, overlaps_xy
    )

    logger.info("Interpolating missing offsets: ")
    logger.info(f"\tNumber of Inf values in coarse_offsets_x: {np.sum(np.isinf(coarse_offsets_x))}")
    logger.info(f"\tNumber of Inf values in coarse_offsets_y: {np.sum(np.isinf(coarse_offsets_x))}")
    
    coarse_offsets_x = stitch_rigid.interpolate_missing_offsets(coarse_offsets_x, -1)
    coarse_offsets_y = stitch_rigid.interpolate_missing_offsets(coarse_offsets_y, -2)

    assert np.inf not in coarse_offsets_x
    assert np.inf not in coarse_offsets_y

    logger.info("optimize_coarse_mesh")

    coarse_mesh = stitch_rigid.optimize_coarse_mesh(
        coarse_offsets_x, coarse_offsets_y
    )

    return np.squeeze(coarse_offsets_x), np.squeeze(coarse_offsets_y), coarse_mesh

def cleanup_flow_fields(fine_x, fine_y):
    logger.info("Cleaning up flow fields")
    kwargs = {
        "min_peak_ratio": 1.4,
        "min_peak_sharpness": 1.4,
        "max_deviation": 5,
        "max_magnitude": 0,
    }
    fine_x = {
        k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
        for k, v in fine_y.items()
    }

    kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
    fine_x = {
        k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[
            :, 0, :, :
        ]
        for k, v in fine_x.items()
    }
    fine_y = {
        k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[
            :, 0, :, :
        ]
        for k, v in fine_y.items()
    }
    return fine_x, fine_y

@with_timer
def compute_flow_maps(coarse_offsets_x, coarse_offsets_y, tile_map, stride):
    # The stride (in pixels) specifies the resolution at which to compute the flow
    # fields between tile pairs. This is the same as the resolution at which the
    # mesh is later optimized. The more deformed the tiles initially are, the lower
    # the stride needs to be to get good stitching results.

    logger.info("Computing flow maps")

    logger.info("compute_flow_map x")
    fine_x, offsets_x = stitch_elastic.compute_flow_map(
        tile_map, coarse_offsets_x, 0, stride=(stride, stride)
    )

    logger.info("compute_flow_map y")
    fine_y, offsets_y = stitch_elastic.compute_flow_map(
        tile_map, coarse_offsets_y, 1, stride=(stride, stride)
    )

    # Clean up the flow fields.
    fine_x, fine_y = cleanup_flow_fields(fine_x, fine_y)

    return fine_x, fine_y, offsets_x, offsets_y

@with_timer
def run_mesh_solver(coarse_offsets_x, coarse_offsets_y, coarse_mesh, fine_x, fine_y, offsets_x, offsets_y, tile_shape, tile_coords, stride):
    logger.info("Preparing data for mesh solver")

    fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
        (coarse_offsets_x, fine_x, offsets_x),
        (coarse_offsets_y, fine_y, offsets_y),
        tile_coords,
        coarse_mesh[:, 0, ...],
        stride=(stride, stride),
        tile_shape=tile_shape,
    )

    # Convert flow maps to dtype=float32
    fx = fx.astype(np.float32)
    fy = fx.astype(np.float32)

    # If we have multiple devices, use them by sharding the mesh, flows, and neighborhood
    # data (by tile dimension) accross devices.
    n_devices = len(jax.local_devices())
    if n_devices > 1:
        logger.info(f"Using {n_devices} devices for mesh solver")
        from jax.experimental import mesh_utils
        from jax.sharding import PositionalSharding
        from jax.experimental.shard_map import shard_map
        devices = mesh_utils.create_device_mesh((n_devices,))
        sharding = PositionalSharding(devices)
        
        # Figure how much we need to pad our number of tiles to be divisible by number of 
        # devices.
        n_tiles = x.shape[1]
        remainder = n_tiles % n_devices
        if remainder > 0:
            n_tiles_to_pad = n_devices - remainder
            x = np.pad(x, ((0, 0), (0, n_tiles_to_pad), (0, 0), (0,0)), mode='constant', constant_values=0.0)
            fx = np.pad(fx, ((0, 0), (0, n_tiles_to_pad), (0, 0), (0,0)), mode='constant', constant_values=np.nan)
            fy = np.pad(fy, ((0, 0), (0, n_tiles_to_pad), (0, 0), (0,0)), mode='constant', constant_values=np.nan)
            
            # Pad with -1, this will cause mesh update to ignore these tiles when updating the mesh
            nbors = np.pad(nbors, ((0, n_tiles_to_pad), (0, 0), (0, 0)), mode='constant', constant_values=-1)

        else:
            n_tiles_to_pad = 0
            
        x = jax.device_put(x, sharding.reshape(1,n_devices,1,1))
        fx = jax.device_put(fx, sharding.reshape(1,n_devices,1,1))
        fy = jax.device_put(fy, sharding.reshape(1,n_devices,1,1))
        nbors = jax.device_put(nbors, sharding.reshape(n_devices,1,1))

        @jax.jit
        def prev_fn(x, fx, fy, nbors):    
            target_fn = ft.partial(
                stitch_elastic.compute_target_mesh,
                x=x,
                fx=fx,
                fy=fy,
                stride=(stride, stride),
            )
            
            x = jax.lax.with_sharding_constraint(jax.vmap(target_fn)(nbors), sharding.reshape((n_devices,1,1,1)))
            return jnp.transpose(x, [1, 0, 2, 3])
        prev_fn_kwargs = {'fx': fx, 'fy': fy, 'nbors': nbors}
        
    else:
        if x.shape[1] < 4000:
            @jax.jit
            def prev_fn(x, fx, fy, nbors):
                target_fn = ft.partial(
                    stitch_elastic.compute_target_mesh,
                    x=x,
                    fx=fx,
                    fy=fy,
                    stride=(stride, stride),
                )
                x = jax.vmap(target_fn)(nbors)
                return jnp.transpose(x, [1, 0, 2, 3])
        else:
            logger.info("Attempting to use chunked version of prev_fn to because of large number of tiles.")
            @jax.jit
            def prev_fn(x, fx, fy, nbors):
                """A chunked version of the prev mesh function that can handle large arrays"""
                target_fn = ft.partial(
                    stitch_elastic.compute_target_mesh,
                    x=x,
                    fx=fx,
                    fy=fy,
                    stride=(stride, stride),
                )
                tv_fn = jax.vmap(target_fn)
                
                chunk_size = int(x.shape[1] / 4)
                b1 = tv_fn(nbors[0:chunk_size, :, :])
                b2 = tv_fn(nbors[chunk_size:(2*chunk_size), :, :])
                b3 = tv_fn(nbors[(2*chunk_size):(3*chunk_size), :, :])
                b4 = tv_fn(nbors[(3*chunk_size):, :, :])
                x = jnp.concatenate((b1, b2, b3, b4), axis=0)

                return jnp.transpose(x, [1, 0, 2, 3])

    prev_fn_kwargs = {'fx': fx, 'fy': fy, 'nbors': nbors}

    # These detault settings are expect to work well in most configurations. Perhaps
    # the most salient parameter is the elasticity ratio k0 / k. The larger it gets,
    # the more the tiles will be allowed to deform to match their neighbors (in which
    # case you might want use aggressive flow filtering to ensure that there are no
    # inaccurate flow vectors). Lower ratios will reduce deformation, which, depending
    # on the initial state of the tiles, might result in visible seams.
    mesh_integration_config = mesh.IntegrationConfig(
        dt=0.001,
        gamma=0.0,
        k0=0.01,
        k=0.1,
        stride=stride,
        num_iters=1000,
        max_iters=100000,
        stop_v_max=0.03,
        dt_max=100,
        prefer_orig_order=True,
        start_cap=0.1,
        final_cap=10.0,
        fire=True,
        remove_drift=True,
    )

    logger.info("Running mesh solver")

    # Turn on logging for the mesh solver
    from absl import logging as logging_absl
    logging_absl.set_verbosity('info')

    # jax.profiler.start_trace("logs/tensorboard_logdir")

    x, ekin, t, v = mesh.relax_mesh(
        x, None, mesh_integration_config, 
        prev_fn=prev_fn, prev_fn_kwargs=prev_fn_kwargs,  
    )

    # x.block_until_ready()
    # jax.profiler.stop_trace()

    # Set the level back to warning
    logging_absl.set_verbosity('warning')
  
    logger.info(f"Mesh solver finished after {t} iterations")

    # Unpack meshes into a dictionary.
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    meshes = {idx_to_key[i]: np.array(x[:, i : i + 1 :, :]) for i in range(x.shape[1]) if i in idx_to_key}

    return meshes, v

@with_timer
def warp_and_render_tiles(tile_map, meshes, stride):
    logger.info("Warping and rendering the stitched tiles")
    stitched, mask = warp.render_tiles(
        tile_map, meshes, stride=(stride, stride), parallelism=usable_cpu_count()
    )
    return stitched, mask


@with_timer
def send_to_cloudvolume(stitched, stitched_filename):
    stitched = stitched[..., np.newaxis]
    x,y,z = stitched.shape
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='image',
        data_type='uint8',
        encoding='raw',
        voxel_offset=[0,0,0],
        resolution=[4,4,40],
        volume_size=[x,y,z],
        chunk_size=[2048, 2048, 1]
    )
    vol = CloudVolume(f'{CLOUDVOLUME_PATH}/{stitched_filename}', info=info)
    vol.commit_info()
    bbox = np.s_[0:x, 0:y, 0:z]
    vol[bbox] = stitched

@with_timer
def downsample_with_igneous(layer_path):
    tq = LocalTaskQueue(parallel=16)
    tasks = create_downsampling_tasks(layer_path)
    tq.insert(tasks)
    tq.execute()


def stitch(section_path: str, x: int, y: int, dx: int, dy: int, output_dir: str, no_render: bool = False, no_upload: bool = False, render_tiff: bool = False):

    logger.info(f"Number of CPU cores: {usable_cpu_count()}")
    logger.info(f"Number of GPU(s): {len(jax.devices())}")
    
    # these should usually work, assuming the section_path has 
    reel = re.search(r'reel(\d+)', section_path).group(1)
    blade = re.search(r'blade(\d+)', section_path).group(1)
    section = re.search(r's(\d+)', section_path).group(1)
    datestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    subset_string = ""
    if dx is not None and dy is not None:
        subset_string = f"_subset_x{x}_y{y}_dx{dx}_dy{dy}"
    
    if output_dir is None:
        stitched_filename = f"reel{reel}_blade{blade}_s{section}_{datestamp}{subset_string}"
        save_path_root = "./"
        save_path = f"{save_path_root}/{stitched_filename}"
    else:
        stitched_filename = pathlib.Path(output_dir).name
        save_path = output_dir

    os.makedirs(save_path, exist_ok=True)

    # Create a wrapper so we can run steps of the pipeline and cache results in
    # pickle files.
    run_step = partial(run_with_cache, pathlib.Path(save_path))

    # Get the supertile map and generate the tile_id_map from stage_positions.csv
    supertile_map = generate_supertile_map_for_section(section_path)
 
    if dx is not None and dy is not None:
        supertile_map = supertile_map[x : x + dx, y : y + dy]
    tile_id_map = generate_tile_id_map(supertile_map)

    ## These are the 5 time-consuming steps of the stitching process

    # 1) Load the tiles from disk
    tile_map = load_tiles(tile_id_map, section_path)

    # Save the tile_map metadata, for the mesh solver step, we don't actually
    # need to load the tile map, we just need its dimensions. This can save 
    # time if we are running only this step.
    save_pickle({"tile_coords": list(tile_map.keys()), 
                 "tile_shape": next(iter(tile_map.values())).shape}, 
                 str(pathlib.Path(save_path) / 'tile_map_meta.pkl')) 

    # 2) Compute the coarse tile positions and the initial mesh.
    # if this fails we just have to reload the tiles, which is not a big deal
    
    coarse_offsets_x, coarse_offsets_y, coarse_mesh = run_step(
        compute_coarse_tile_positions, tile_space=tile_id_map.shape, tile_map=tile_map
    )
    
    # Resolution for flow field computation and mesh optimization.
    STRIDE = 30

    # 3) Compute the flow maps between tile pairs.
    # if this or any subsequent step fails, we want to save the work we've done so far
    fine_x, fine_y, offsets_x, offsets_y = run_step(
        compute_flow_maps, coarse_offsets_x, coarse_offsets_y, tile_map, STRIDE
    )

    # 4) Run the mesh solver.
    tile_map_meta = load_pickle(str(pathlib.Path(save_path) / 'tile_map_meta.pkl'))
    tile_shape = tile_map_meta['tile_shape']
    tile_coords = tile_map_meta['tile_coords']

    meshes = run_step(
        run_mesh_solver,
        coarse_offsets_x,
        coarse_offsets_y,
        coarse_mesh,
        fine_x,
        fine_y,
        offsets_x,
        offsets_y,
        tile_shape,
        tile_coords,
        STRIDE,
    )

    if no_render:
        logger.info("Stiching completed sucessfully, skipping rendering of stitched images. "
                    "Run without --no_render option to finnish")
        return

    # 5) Warp the tiles into a single image.
    out_image_path = f"{save_path}/{stitched_filename}.npy"
    if not pathlib.Path(out_image_path).exists():
        
        stitched, _ = warp_and_render_tiles(tile_map, meshes, STRIDE)
        
        logger.info("Saving stitched image")
        np.save(out_image_path, stitched)

    else:
        logger.info("Loading stitched image from disk")
        stitched = np.load(out_image_path)

    tiff_path = f"{save_path}/{stitched_filename}.tiff"
    if render_tiff and not pathlib.Path(tiff_path).exists():
        import tifffile
        logger.info(f"Saving stitched image as tiff: {tiff_path}")
        tifffile.imwrite(tiff_path, stitched)
        
    if no_upload:
        logger.info("Skipping uploading of image to cloudvolume")
        return

    logger.info("sending to cloudvolume")

    # if this fails for some reason, we fall back to saving to disk
    try:
        send_to_cloudvolume(stitched, stitched_filename)
    except Exception as e:
        logger.info("Sending to cloudvolume failed.")
        raise e

    # downsample with igneous
    layer_path = f"{CLOUDVOLUME_PATH}/{stitched_filename}"
    logger.info("downsampling with igneous")
    downsample_with_igneous(layer_path)

    logger.info(f"precomputed://https://s3-hpcrc.rc.princeton.edu/bucket-test/{stitched_filename}")
    logger.info("Stitching completed successfully and result saved.")

@click.command()
@click.argument('section_path', type=click.Path(exists=True))
@click.option('--x', type=int, default=0, help='First x position of the supertile subset')
@click.option('--y', type=int, default=0, help='First y position of the supertile subset')
@click.option('--dx', type=int, help='Width of the supertile subset')
@click.option('--dy', type=int, help='Height of the supertile subset')
@click.option('--no_render', is_flag=True, show_default=True, default=False, help='Skip rendering.')
@click.option('--no_upload', is_flag=True, show_default=True, default=False, help='Skip uploading to cloudvolume.')
@click.option('--output_dir', type=str, help="Directory to save results.", default=None)
@click.option('--render_tiff', is_flag=True, show_default=True, default=False, help='Save the stictched result image to disk as tiff.')
def cli(section_path, x, y, dx, dy, output_dir, no_render, no_upload, render_tiff):
    """
    Stitch a section from disk at the specified SECTION_PATH.
    """
    logger.info(f"Stitching section at {section_path}")
    if dx is not None and dy is not None:
        logger.info(f"Subset: x={x}, y={y}, dx={dx}, dy={dy}")

    total_start = time()

    stitch(section_path=section_path, x=x,y= y, dx=dx, dy=dy, 
           output_dir=output_dir, no_render=no_render, no_upload=no_upload, render_tiff=render_tiff) 
    total_end = time()
    elapsed_time = total_end - total_start
    logger.info(f"Total time elapsed: {elapsed_time:.0f} seconds.")

if __name__ == "__main__":
    cli()