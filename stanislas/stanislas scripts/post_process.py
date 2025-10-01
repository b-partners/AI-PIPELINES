import json, ijson, os
import math
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from shapely.strtree import STRtree

# Helper Functions

def pixel_to_mercator(px, py, zoom, tile_size):
    """Convert global pixel coordinates to EPSG:3857."""
    px, py = float(px), float(py)   # ✅ ensure floats
    initial_resolution = 2 * math.pi * 6378137 / tile_size
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2 ** zoom)
    mx = px * res - origin_shift
    my = origin_shift - py * res
    return mx, my

def vgg_to_geojson_stream(vgg_json_path, zoom, tile_size, output_path, epsg=3857, save_inter=True):

    geometries, classes, scores = [], [], []

    with open(vgg_json_path, 'r') as f:
        parser = ijson.kvitems(f, '')  # stream over top-level keys (filenames)

        for filename, data in parser:
            
            try:
                tile_x, tile_y = map(int, filename.replace(".jpg", "").split("_"))
                year = 22
            except ValueError:
                continue

            for region in data["regions"].values():
                shape_attr = region["shape_attributes"]
                region_attr = region.get("region_attributes", {})
                cls = region_attr.get("label", "")
                score = region_attr.get('confidence', 0.)

                xs = shape_attr["all_points_x"]
                ys = shape_attr["all_points_y"]

                x_global = [tile_x * tile_size + x for x in xs]
                y_global = [tile_y * tile_size + y for y in ys]

                coords_merc = [pixel_to_mercator(px, py, zoom, tile_size) 
                               for px, py in zip(x_global, y_global)]
                
                if len(coords_merc) >= 3:
                    poly = Polygon(coords_merc)
                    if poly.is_valid:
                        geometries.append(poly)
                        classes.append(cls)
                        scores.append(score)

    gdf = gpd.GeoDataFrame(
        {"label": classes, "geometry": geometries, "scores": scores},
        crs=f"EPSG:{epsg}"
    )
    
    if save_inter:
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Saved to: {output_path}")
    
    return gdf

def polygon_iou(poly1, poly2):
    """Compute IoU between two polygons."""
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0


def nms_polygons(gdf, iou_thresh=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on polygons in a GeoDataFrame.
    Keeps the highest scoring polygon in each overlapping cluster.
    """
    geoms = list(gdf.geometry)
    scores = gdf["scores"].values
    labels = gdf["label"].values

    # Sort polygons by score (high → low)
    order = scores.argsort()[::-1]

    keep = []
    suppressed = set()

    tree = STRtree(geoms)

    for i in order:
        if i in suppressed:
            continue

        # Keep the highest score polygon
        keep.append(i)

        # Query spatial index for potential overlaps
        candidates = tree.query(geoms[i])

        for j in candidates:
            if j == i or j in suppressed:
                continue
            if labels[i] != labels[j]:  
                # Optionally: only suppress if same class
                continue
            iou = polygon_iou(geoms[i], geoms[j])
            if iou >= iou_thresh:
                suppressed.add(j)

    return gdf.iloc[keep].reset_index(drop=True)

def nms_geojson(gdf, output_geojson, iou_thresh=0.5, save_inter=True):

    # Ensure needed columns
    assert "label" in gdf.columns and "scores" in gdf.columns, \
        "GeoJSON must have 'label' and 'scores'"

    # Apply polygon NMS
    cleaned = nms_polygons(gdf, iou_thresh=iou_thresh)

    # Save result
    if save_inter:
        cleaned.to_file(output_geojson, driver="GeoJSON")
        print(f"Saved NMS-cleaned GeoJSON -> {output_geojson}")
    
    return cleaned


def polygon_filter(gdf, inter_thresh=0.8):
    """
    Remove small polygons that are mostly covered by larger polygons using STRtree.
    Compatible with Shapely >= 2.0 (query returns indices).
    """
    gdf = gdf.copy()
    gdf["area"] = gdf.geometry.area

    # Sort polygons by area (largest → smallest)
    gdf_sorted = gdf.sort_values("area", ascending=False).reset_index(drop=True)

    geoms = list(gdf_sorted.geometry)
    tree = STRtree(geoms)

    to_drop = set()

    for i, big in gdf_sorted.iterrows():
        if i in to_drop:
            continue

        # Query candidates intersecting with the big polygon (returns indices in Shapely 2.x)
        cand_idx = tree.query(big.geometry)

        for j in cand_idx:
            if j <= i or j in to_drop:
                continue

            small = gdf_sorted.iloc[j]
            inter = big.geometry.intersection(small.geometry)
            if not inter.is_empty:
                inter_ratio = inter.area / small.area
                if inter_ratio > inter_thresh:
                    to_drop.add(j)

    return gdf_sorted.drop(index=list(to_drop)).reset_index(drop=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="post process the json issued from inference.py of place stanislas")
    parser.add_argument("--in-vgg-path", help="the path of the vgg json file issued from the inference.py", required=True)
    parser.add_argument("--zoom", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--out-geojson", help="the path of the geoJson file after post processing", required=True)
    parser.add_argument("--post-nms", help="apply post NMS to the geoJson", type=bool, default=True)
    parser.add_argument("--iou", help="IoU threshold if post-NMS", type=float, default=0.1)
    parser.add_argument("--filter", help="filter geoJson from small polygons", type=bool, default=True)
    parser.add_argument("--ios", help="IoS threshold for filtering", type=float, default=0.8)
    parser.add_argument("--save-inter", help="save intermed geoJsons", type=bool, default=True)
    
    args = parser.parse_args()
    
    if args.save_inter:
        b_fn = os.path.basename(args.out_geojson)
        
        raw_fn = f"raw_{b_fn}"
        raw_fp = args.out_geojson.replace(b_fn, raw_fn)
        
        nms_fn = f"nms_{b_fn}"
        nms_fp = args.out_geojson.replace(b_fn, nms_fn)
    
    gdf = vgg_to_geojson_stream(
        vgg_json_path=args.in_vgg_path,
        zoom = args.zoom,
        tile_size=args.imgsz,
        output_path=raw_fp,
        save_inter=args.save_inter
    )
    
    if args.post_nms:
        gdf = nms_geojson(
            gdf,
            nms_fp,
            iou_thresh=args.iou,
            save_inter=args.save_inter
        )
        
    if args.filter:
        gdf = polygon_filter(gdf, inter_thresh=args.ios)
    
    gdf.to_file(args.out_geojson, driver="GeoJSON")  
    print(f"saved final geojson to: {args.out_geojson}")
        
