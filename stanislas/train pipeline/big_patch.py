import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

def load_big_patch(folder, fname, patch_size=256, overlap=0):
    """
    Given a tile filename (x_y.jpg), load the tile with its neighbors 
    to build a larger patch including overlap context.

    Parameters
    ----------
    folder : str
        Path to folder containing tiles.
    fname : str
        Tile filename like "12_8.jpg".
    patch_size : int
        Size of each tile (default 256).
    overlap : int
        Context overlap (default 0).
        Example: overlap=64 → returns (256+2*64)x(256+2*64) patch.

    Returns
    -------
    big_patch : np.ndarray
        Image containing the tile + context.
    center_coords : tuple
        (row_start, row_end, col_start, col_end) for the center tile inside big_patch.
    """

    # parse x, y from filename
    pattern = re.compile(r"(\d+)_(\d+)\.jpg")
    match = pattern.match(fname)
    if not match:
        raise ValueError(f"Filename {fname} does not match x_y.jpg pattern")
    x, y = map(int, match.groups())

    # compute how many extra tiles needed
    extra = int(np.ceil(overlap / patch_size))  # usually 1 if overlap < patch_size

    # size of final big patch
    size = patch_size * 3
    big_patch = np.zeros((size, size, 3), dtype=np.uint8)

    # load neighborhood covering needed overlap
    for dx in range(-extra, extra + 1):
        for dy in range(-extra, extra + 1):
            nx, ny = x + dx, y + dy
            nfname = f"{nx}_{ny}.jpg"
            path = os.path.join(folder, nfname)
            if os.path.exists(path):
                tile = cv2.imread(path)
            else:
                tile = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

            # compute where this neighbor falls inside big_patch
            row_start = (dy + extra) * patch_size
            col_start = (dx + extra) * patch_size
            big_patch[row_start:row_start+patch_size, col_start:col_start+patch_size] = tile

    # compute crop coordinates for center tile inside big_patch
    the_patch = big_patch[overlap:-overlap, overlap :-overlap]
    
    return the_patch

if __name__ == "__main__":
    source = 'Fusion_data_Place_Stanislas_256X256_Z25'
    target = 'big_images'

    patch_size = 256
    overlap = 128
    
    for fname in tqdm(os.listdir(source), desc="loading big patches", unit="image(s)"):
        big_patch = load_big_patch(source, fname, patch_size=patch_size, overlap=overlap)
        cv2.imwrite(f"{target}/{fname}", big_patch)