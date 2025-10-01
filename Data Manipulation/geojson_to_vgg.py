import json
import math
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import transform

def mercator_to_pixel(x, y, zoom, tile_size):
    initial_resolution = 2 * math.pi * 6378137 / tile_size
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2 ** zoom)
    px = (x + origin_shift) / res
    py = (origin_shift - y) / res
    return px, py

def pixel_to_mercator_bounds(tile_x, tile_y, zoom, tile_size):
    """Get EPSG:3857 bounding box for a tile."""
    initial_resolution = 2 * math.pi * 6378137 / tile_size
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2 ** zoom)

    minx = tile_x * tile_size * res - origin_shift
    maxx = (tile_x + 1) * tile_size * res - origin_shift
    maxy = origin_shift - tile_y * tile_size * res
    miny = origin_shift - (tile_y + 1) * tile_size * res
    return box(minx, miny, maxx, maxy)

def convert_geojson_to_vgg_multitile(geojson_path, zoom, tile_size, class_property="class"):
    gdf = gpd.read_file(geojson_path).to_crs(epsg=3857)
    # gdf = gdf.to_crs(3857)
    vgg_dict = {}
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        if geometry.is_empty or not geometry.is_valid:
            continue

        class_label = row.get(class_property, "")
        year = row.get('year', None)
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        else:
            continue

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            # Get pixel bounds
            px_min, py_max = mercator_to_pixel(minx, miny, zoom, tile_size)
            px_max, py_min = mercator_to_pixel(maxx, maxy, zoom, tile_size)

            tile_x_min = int(px_min // tile_size)
            tile_x_max = int(px_max // tile_size)
            tile_y_min = int(py_min // tile_size)
            tile_y_max = int(py_max // tile_size)

            for tx in range(tile_x_min, tile_x_max + 1):
                for ty in range(tile_y_min, tile_y_max + 1):
                    tile_bounds = pixel_to_mercator_bounds(tx, ty, zoom, tile_size)
                    clipped = poly.intersection(tile_bounds)
                    if clipped.is_empty:
                        continue

                    clipped_polys = [clipped] if isinstance(clipped, Polygon) else clipped.geoms

                    for part in clipped_polys:
                        if part.is_empty:
                            continue

                        x, y = part.exterior.coords.xy
                        pixel_coords = [mercator_to_pixel(x[i], y[i], zoom, tile_size) for i in range(len(x))]

                        all_x = [pt[0] - tx * tile_size for pt in pixel_coords]
                        all_y = [pt[1] - ty * tile_size for pt in pixel_coords]

                        region = {
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": [int(x) for x in all_x],
                                "all_points_y": [int(y) for y in all_y],
                            },
                            "region_attributes": {
                                "label": class_label
                            }
                        }
                        if year:
                            tile_name = f"{tx}_{ty}.jpg"
                        else:
                            tile_name = f"{tx}_{ty}.jpg"
                            
                        if tile_name not in vgg_dict:
                            vgg_dict[tile_name] = {
                                "filename": tile_name,
                                "size": 0,
                                "regions": {"0": region},
                                "file_attributes": {}
                            }
                        else:
                            vgg_dict[tile_name]["regions"][f"{len( vgg_dict[tile_name]['regions'])}"] = region

    return vgg_dict

# Example usage
if __name__ == "__main__":
    geojson_path = "/home/adelb/Documents/Bpartners/Stanislas/big_VGGs/small_state_3857.geojson"
    zoom = 25
    tile_size = 256
    class_property = "label"  # or "label"

    vgg_json = convert_geojson_to_vgg_multitile(geojson_path, zoom, tile_size, class_property)

    with open(f"/home/adelb/Documents/Bpartners/Stanislas/big_VGGs/small_state.json", "w") as f:
        json.dump(vgg_json, f, indent=2)
