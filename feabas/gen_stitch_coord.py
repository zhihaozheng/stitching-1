
import pandas as pd
import numpy as np
import os
import click
import re

TIFF_ROOT = "/scratch/rmorey/mec/tiff_sections/reel1068_blade2"
STITCH_COORDS_DIR = "./stitch_coords"

RESOLUTION = 4.0  # Assuming a resolution of 4.0 nm/pixel
TILE_SIZE = (6000, 6000)  # Size of each tile in pixels
OVERLAP = 2000  # Overlap in pixels

# Generate supertile map from stage_positions.csv
def generate_supertile_map(csv_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Correct the column names by removing leading spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Normalize the stage_x_nm and stage_y_nm values
    df['norm_stage_x'] = df['stage_x_nm'].rank(method='dense').astype(int) - 1
    df['norm_stage_y'] = df['stage_y_nm'].rank(method='dense').astype(int) - 1
    
    # Determine the dimensions of the 2D array
    max_x = df['norm_stage_x'].max()
    max_y = df['norm_stage_y'].max()
    
    # Initialize an array of shape (max_y+1, max_x+1) with None values
    arr = np.full((max_y+1, max_x+1), None)
    
    # Populate the array with tile_id values using numpy's advanced indexing
    arr[df['norm_stage_y'].values, df['norm_stage_x'].values] = df['tile_id'].values
    
    # Reverse the order of the rows in the array
    supertile_map = arr[::-1]
    
    return supertile_map

def generate_tile_id_map(supertile_map):

    # Cricket subtile order
    SUBTILE_MAP = [[6, 7, 8], [5, 0, 1], [4, 3, 2]]

    tile_id_map = []
    for supertile_row in supertile_map:
        for subtile_row in SUBTILE_MAP: 
            current_row = []
            for supertile in supertile_row:
                for subtile in subtile_row:
                    if supertile is not None:
                        current_row.append(f"{supertile:04}_{subtile}")
                    else:
                        current_row.append(None)
            tile_id_map.append(np.array(current_row))
    return np.vstack(tile_id_map)


# Function to calculate the coordinates of each tile
def calculate_tile_coordinates(tile_id_map, tile_size, overlap):
    tile_coordinates = []
    step_size = (tile_size[0] - overlap, tile_size[1] - overlap)  # Step size considering overlap
    for y in range(tile_id_map.shape[0]):
        for x in range(tile_id_map.shape[1]):
            # Calculate the coordinates
            coord_x = x * step_size[0]
            coord_y = y * step_size[1]
            tile_coordinates.append((tile_id_map[y, x], coord_x, coord_y))
    return tile_coordinates

def gen_stitch_coords(root_dir, csv_path, output_dir):
    # Generate supertile map
    supertile_map = generate_supertile_map(csv_path)

    # Generate tile ID map
    tile_id_map = generate_tile_id_map(supertile_map)

    # Calculate the coordinates for each tile
    tile_coordinates = calculate_tile_coordinates(tile_id_map, TILE_SIZE, OVERLAP)

    # File content preparation
    file_content = [
        "{ROOT_DIR}\t" + root_dir,
        "{RESOLUTION}\t" + str(RESOLUTION),
        "{TILE_SIZE}\t" + "\t".join(map(str, TILE_SIZE))
    ]
    file_content.extend([f"tile_{tile_id}.tif\t{coord_x}\t{coord_y}" for tile_id, coord_x, coord_y in tile_coordinates if tile_id is not None])

    # Joining content into a single string
    file_content_str = "\n".join(file_content)
    
    # Search for the pattern in the text
    match = re.search(r"/(s\d+)-", csv_path)

    # Extracted value
    if match:
        section = match.group(1)
    else:
        print('no match found')
        print(csv_path)
        exit()

    # Extract the section name and construct the output file path
    section_name = os.path.basename(csv_path).split('_stage_positions.csv')[0].split('-')[0]
    output_path = os.path.join(output_dir, section + '.txt')

    # Writing the content to the file
    with open(output_path, 'w') as file:
        file.write(file_content_str)

    print(f"Output written to {output_path}")

def run_script_for_all_csv(root_dir, stage_positions_dir, output_dir):
    # List all CSV files in the stage positions directory
    csv_files = [f for f in os.listdir(stage_positions_dir) if f.endswith('_stage_positions.csv')]

    # Iterate through each CSV file
    for csv_file in csv_files:
        # Extract the section name from the CSV file name
        section_name = csv_file.split('_stage_positions.csv')[0]

        # Construct the full paths for the root directory and CSV file
        root_path = os.path.join(root_dir, section_name)
        csv_path = os.path.join(stage_positions_dir, csv_file)

        # Execute the function for each file
        gen_stitch_coords(root_path, csv_path, output_dir)

@click.command
@click.argument('stage_positions_path', type=click.Path(exists=True))
def main(stage_positions_path):

    match = re.search(r"/(s\d+-\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2})/", stage_positions_path)

    # Extracted value
    if match:
        section_root_dir = match.group(1)
    else:
        print('no match found')
        print(stage_positions_path)
        exit()


    gen_stitch_coords(f"{TIFF_ROOT}/{section_root_dir}", stage_positions_path, STITCH_COORDS_DIR)
    
if __name__ == "__main__":
    main()