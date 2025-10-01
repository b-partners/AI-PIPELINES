import json, ijson
import math
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer


def pixel_to_mercator(px, py, zoom, tile_size):
    """Convert global pixel coordinates to EPSG:3857."""
    px, py = float(px), float(py)   # ✅ ensure floats
    initial_resolution = 2 * math.pi * 6378137 / tile_size
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = initial_resolution / (2 ** zoom)
    mx = px * res - origin_shift
    my = origin_shift - py * res
    return mx, my

def vgg_to_geojson(vgg_json_path, zoom, tile_size, output_path, epsg):
    with open(vgg_json_path) as f:
        vgg_data = json.load(f)

    geometries = []
    classes = []
    years = []

    for filename, data in vgg_data.items():
        try:
            tile_x, tile_y = map(int, filename.replace(".jpg", "").split("_"))
            year=22
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        for region in data["regions"].values():
            shape_attr = region["shape_attributes"]
            region_attr = region.get("region_attributes", {})
            cls = region_attr.get("label", "")

            x_local = shape_attr["all_points_x"]
            y_local = shape_attr["all_points_y"]

            # Global pixel coordinates
            x_global = [tile_x * tile_size + x for x in x_local]
            y_global = [tile_y * tile_size + y for y in y_local]

            # Convert to EPSG:3857
            coords_mercator = [pixel_to_mercator(px, py, zoom, tile_size) for px, py in zip(x_global, y_global)]

            # Create polygon
            if len(coords_mercator) >= 3:
                poly = Polygon(coords_mercator)
                if poly.is_valid:
                    geometries.append(poly)
                    classes.append(cls)
                    years.append(year)

    gdf = gpd.GeoDataFrame({"especes": classes, "geometry": geometries, 'year':years}, crs=f"EPSG:{epsg}")
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved to: {output_path}")

def vgg_to_geojson_boxes(vgg_json_path, zoom, tile_size, output_path, epsg):
    """Convert VGG format with rectangular annotations into a GeoJSON of boxes."""
    from PIL import Image
    
    with open(vgg_json_path) as f:
        vgg_data = json.load(f)

    geometries = []
    classes = []
    years = []

    for filename, data in vgg_data.items():
        try:
            year, tile_x, tile_y = map(int, filename.replace(".jpg", "").split("_"))
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        for region in data["regions"].values():
            shape_attr = region["shape_attributes"]
            region_attr = region.get("region_attributes", {})
            cls = region_attr.get("label", "")

            x = shape_attr["x"]
            y = shape_attr["y"]
            w = shape_attr["width"]
            h = shape_attr["height"]

            # Define the 4 corner points of the rectangle in pixel space
            x_vals = [x, x + w, x + w, x]
            y_vals = [y, y, y + h, y + h]

            # Transform to global pixel
            x_global = [tile_x * tile_size + px for px in x_vals]
            y_global = [tile_y * tile_size + py for py in y_vals]

            # Convert to Mercator
            coords_mercator = [pixel_to_mercator(px, py, zoom, tile_size) for px, py in zip(x_global, y_global)]

            poly = Polygon(coords_mercator)
            if poly.is_valid:
                geometries.append(poly)
                classes.append(cls)
                years.append(year)

    gdf = gpd.GeoDataFrame({"especes": classes, "geometry": geometries, "year": years}, crs=f"EPSG:{epsg}")
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved to: {output_path}")


def vgg_to_geojson_stream(vgg_json_path, zoom, tile_size, output_path, epsg):

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
    
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    vgg_to_geojson_stream(
        vgg_json_path="/home/adelb/Downloads/test_sahi_state_clean.json",
        zoom = 25,
        tile_size=256,
        output_path="/home/adelb/Downloads/test_sahi_state_clean.geojson",
        epsg = 3857
    )