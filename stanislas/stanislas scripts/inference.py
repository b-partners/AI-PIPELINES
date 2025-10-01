import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon
import json

# Helper Functions

def load_big_patch(folder, fname, patch_size=256, overlap=128):
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

def get_model(model_path:str, conf:float=0.35, device:str="cpu", imgsz:int=256):
    detection_model = AutoDetectionModel.from_pretrained(
                        model_type="ultralytics",
                        model_path=model_path,
                        confidence_threshold=conf,
                        device=device,  # or 'cuda:0'
                        image_size=imgsz
                    )
    return detection_model

def largest_polygon_xy(segments):
    """
    Given a list of polygons (each polygon is [x0,y0,x1,y1,...]),
    return the polygon with largest area as (all_x, all_y).

    Parameters
    ----------
    segments : list[list[float]]
        List of polygons, each polygon = [x0,y0, x1,y1, ..., xn,yn]

    Returns
    -------
    all_x : list[float]
        X coordinates of largest polygon
    all_y : list[float]
        Y coordinates of largest polygon
    max_area : float
        Area of the largest polygon
    """
    max_area = 0
    all_x, all_y = [], []

    for coords in segments:
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        poly = Polygon(points)

        if poly.area > max_area:
            max_area = poly.area
            x, y = zip(*poly.exterior.coords)  # extract coords from polygon
            all_x, all_y = list(x), list(y)

    return all_x, all_y

def _to_point_array(segments_or_points):
    """
    Accepts:
      - Nx2 array-like -> treated as points
      - Mx4 array-like -> treated as segments [x0,y0,x1,y1] (endpoints aggregated)
    Returns:
      - numpy array of shape (K,2) of unique points (float)
    """
    pts = np.asarray(segments_or_points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("Input must be a 2D array-like of shape (N,2) or (M,4).")

    if pts.shape[1] == 2:
        pts2 = pts
    elif pts.shape[1] == 4:
        # each row: x0,y0,x1,y1 -> collect endpoints
        a = pts[:, :2]
        b = pts[:, 2:4]
        pts2 = np.vstack([a, b])
    else:
        raise ValueError("Input must be Nx2 (points) or Mx4 (segments [x0,y0,x1,y1]).")

    # unique rows to reduce duplicates
    # round slightly to avoid floating uniqueness issues
    pts_rounded = np.round(pts2, decimals=6)
    # use view hack to unique rows
    uniq = np.unique(pts_rounded.view([('', pts_rounded.dtype)] * 2)).view(pts_rounded.dtype).reshape(-1, 2)
    # keep as float (original)
    # But preserve original ordering by matching (safe enough to return uniq)
    return uniq.astype(float)

def oriented_bbox(points_or_segments, method='opencv'):
    """
    Compute oriented bounding box (minimum-area rectangle) from points or segments.

    Args:
      points_or_segments: array-like, either
         - Nx2 points: [[x,y], ...]
         - Mx4 segments: [[x0,y0,x1,y1], ...]
      method: 'opencv' (default) or 'shapely' fallback.

    Returns:
      dict with keys:
        - 'center': (cx, cy)
        - 'width': float (box width)
        - 'height': float (box height)
        - 'angle': float (degrees). See note below for convention.
        - 'corners': np.array shape (4,2) in order (clockwise or cv2 order)
        - 'polygon': list of corners as [(x,y), ...] (same order)
    """
    pts = _to_point_array(points_or_segments)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 unique points to compute an oriented bounding box.")

    # cv2 expects float32
    arr = pts.astype(np.float32)
    r = cv2.minAreaRect(arr)  # ((cx,cy), (w,h), angle)
    (cx, cy), (w, h), angle = r
    # get box corners
    box = cv2.boxPoints(r)  # returns 4x2 float32
    # boxPoints returns in order, but not guaranteed CW/CCW consistent; it's fine for usage
    corners = box.astype(float)
    # normalize angle: keep CV angle but also provide angle_longer which is angle of longer side in [0,180)
    if w >= h:
        angle_longer = angle % 180
    else:
        # swap to make width the longer side
        w, h = h, w
        angle_longer = (angle + 90) % 180

    return {
        'center': (float(cx), float(cy)),
        'width': float(w),
        'height': float(h),
        'angle': float(angle_longer),   # angle of longer side (deg, in [0,180))
        'raw_angle_cv2': float(angle),  # raw cv2 angle (-90..0]
        'corners': corners,
        'polygon': np.array([tuple(map(int, c)) for c in corners])
    }

def sahi_to_obb_vgg(segments):
    all_x, all_y = largest_polygon_xy(segments)
    pts = np.array(list(zip(all_x, all_y)), dtype = 'int')
    obb = oriented_bbox(pts)
    poly = obb['polygon']
    Xs, Ys = poly.T
    
    return Xs, Ys, obb

def int_and_offset(lst, offset):
    """
    Convert numbers in list to int and subtract offset.
    """
    return [int(v) - offset for v in lst]


def is_valid_stone(obb, angle_only=True):
    angle, height, width = obb['raw_angle_cv2'], obb['height'] , obb['width']
    if angle_only and 15 <= angle <= 75:
        return True
    
    if 70 <= width <= 100 and 50 <= height <= 80 and 15 <= angle <= 75:
        return True
    
    return False

def sahi_result_to_vgg_region(sahi_res, offset = 128, angle_only=True):
    label = sahi_res.category.name
    
    Xs, Ys, obb = sahi_to_obb_vgg(sahi_res.mask.segmentation)
    Xs = int_and_offset(Xs, offset)
    Ys = int_and_offset(Ys, offset)
    
    confidence = sahi_res.score.value
    
    
    if not is_valid_stone(obb, angle_only=angle_only):
        return None
    
    return {
        'shape_attributes':
            {
                'name': 'polygon',
                'all_points_x': Xs,
                'all_points_y': Ys
            },
        'region_attributes':
            {
                'label': label,
                'confidence': confidence    
            },
    }
    
def sahi_to_vgg(sahi_results, fname, offset = 128, angle_only = True):
    vgg ={
                'filename': fname,
                'size': 0,
                'regions': {},
        }
    
    regions = {}
    
    for i, object in enumerate(sahi_results.object_prediction_list):
        region = sahi_result_to_vgg_region(object, offset, angle_only=angle_only)
        if region is not None:
            regions[str(i)] = region
        
    vgg['regions'].update(regions)
    
    return vgg

def process_folder(detection_model, folder, vgg_out_path, patch_size = 256, overlap=128, process= "NMS", metric="IOU", thresh=0.2, batch_size=128, angle_only=True):
    
    offset = patch_size - overlap
    vgg = {}
    fnames = os.listdir(folder)
    for fname in fnames:
        big_patch = load_big_patch(folder, fname, patch_size=patch_size, overlap=overlap)
        results = get_sliced_prediction(
                                        big_patch,
                                        detection_model,
                                        slice_height=patch_size,
                                        slice_width=patch_size,
                                        overlap_height_ratio=overlap/patch_size,
                                        overlap_width_ratio=overlap/patch_size,
                                        postprocess_type = process,
                                        postprocess_match_metric = metric,
                                        postprocess_match_threshold = thresh,
                                    )
        vgg[fname] = sahi_to_vgg(results, fname, offset, angle_only=angle_only)
        
    with open(vgg_out_path, 'w') as f:
        json.dump(vgg, f)
    
    print(f"{folder} : Done | treated {len(os.listdir(folder))} images\nResult saved to: {vgg_out_path}")
    
    
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Inference on place Stanislas")
    parser.add_argument("--model-path", help="yolo model path", required=True)
    parser.add_argument("--source", help="folder containing images", required=True)
    parser.add_argument("--out-vgg-path", type=str, default='./tmp.json', help="the vgg json file path of the results")
    parser.add_argument("--imgsz", type=int, default=256, help="image size for inference")
    parser.add_argument("--overlap", type=int, default=128, help="Fixed overlap value in pixels")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--process", choices=["NMS", "NMM", "GREEDYNMM"], default='NMS')
    parser.add_argument("--metric", choices=['IOU', 'IOS'], default='IOU')
    parser.add_argument("--device", choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument("--metric-thresh", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--angle_only", type=bool, default=True)
    args = parser.parse_args()
    
   
    
    model = get_model(args.model_path, args.conf, args.device, args.imgsz)
    
    process_folder(model, args.source, args.out_vgg_path, args.imgsz, args.overlap, args.process, args.metric, args.metric_thresh, args.batch_size, args.angle_only)
    
    
    
    
    