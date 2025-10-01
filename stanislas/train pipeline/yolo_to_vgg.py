import os
import json
import cv2
import glob

_has_cv2 = True

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


def poly_to_obb(all_x, all_y):
    pts = np.array(list(zip(all_x, all_y)), dtype = 'int')
    obb = oriented_bbox(pts)
    return obb

def yolo_results_to_vgg(results, class_mapping, output_file, offset=128):
    """
    Convert YOLOv8 segmentation results to VGG JSON format with confidence scores.

    Args:
        results (list): YOLOv8 results containing polygons, class information, and scores.
        class_mapping (dict): Mapping of class indices to class labels.
        output_file (str): Path to save the VGG JSON file.
    """
    vgg_data = {}

    for result in results:
        if not hasattr(result, "path") or not os.path.exists(result.path):
            print(f"Warning: Missing or invalid path for result {result}. Skipping.")
            continue

        image_filename = os.path.basename(result.path)
        vgg_entry = {
            "fileref": "",
            "size": os.path.getsize(result.path),
            "filename": image_filename,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {}
        }

        # Check if masks and boxes are available
        if result.masks is None or result.boxes is None:
            print(f"No masks or boxes detected for image: {image_filename}.")
            vgg_data[image_filename] = vgg_entry
            continue

        # Iterate over each detection
        region_index = 0
        for segment, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
            cls_index = int(cls)  # YOLO class indices start from 0
            label = class_mapping.get(cls_index, f"class_{cls_index}")
            confidence = float(conf)  # Convert confidence score to float

            all_points_x = [x - offset for x in segment[:, 0].tolist()]
            all_points_y = [x - offset for x in segment[:, 1].tolist()]
            obb = poly_to_obb(all_points_x, all_points_y)
            vgg_entry["regions"][str(region_index)] = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "obb":obb,
                "region_attributes": {
                    "label": label,
                    "confidence": confidence
                }
            }
            region_index += 1

        vgg_data[image_filename] = vgg_entry

    # Save to output file
    with open(output_file, "w") as json_file:
        json.dump(vgg_data, json_file, indent=4)
    print(f"VGG annotations with confidence scores saved to {output_file}")

