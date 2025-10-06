import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import os

def clean_gdf(gdf: gpd.GeoDataFrame, target_crs="EPSG:3857") -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.buffer(0)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def filter_comparison_by_intersection(reference_gdf, comparison_gdf, inter_frac=0.5):
    reference_prepared = [prep(geom) for geom in reference_gdf.geometry]
    kept_geoms = []

    for comp_geom in comparison_gdf.geometry:
        keep = any(
            prep_geom.intersects(comp_geom) and
            (comp_geom.intersection(ref_geom).area / ref_geom.area > inter_frac)
            for prep_geom, ref_geom in zip(reference_prepared, reference_gdf.geometry)
            if ref_geom.area > 0
        )
        if keep:
            kept_geoms.append(comp_geom)

    return gpd.GeoDataFrame(geometry=kept_geoms, crs=comparison_gdf.crs)

def cluster_and_bbox_flexible_merged(
    gdf,
    area_thresh=10.0,
    max_distance=20.0,
    num_bati=3,
    minimal_bbox_area=100.0
):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.area >= area_thresh].reset_index(drop=True)
    if gdf.empty:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=gdf.crs)

    gdf['cluster'] = -1
    cluster_id = 0
    remaining = gdf.copy()

    for min_samples in range(num_bati, 0, -1):
        if remaining.empty:
            break

        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in remaining.geometry])
        clustering = DBSCAN(eps=max_distance, min_samples=min_samples).fit(centroids)

        labels = clustering.labels_
        remaining['temp_cluster'] = labels

        for label in set(labels):
            if label == -1:
                continue
            mask = remaining['temp_cluster'] == label
            if mask.sum() >= min_samples:
                gdf.loc[remaining[mask].index, 'cluster'] = cluster_id
                cluster_id += 1

        remaining = remaining[remaining['temp_cluster'] == -1].drop(columns='temp_cluster')

    clustered = gdf[gdf['cluster'] != -1]
    cluster_bboxes = [box(*unary_union(group.geometry).bounds) for _, group in clustered.groupby('cluster')]

    unclustered = gdf[gdf['cluster'] == -1]
    large_noise_bboxes = [box(*geom.bounds) for geom in unclustered.geometry if geom.area >= 2 * area_thresh]

    all_bboxes = cluster_bboxes + large_noise_bboxes
    merged_union = unary_union(all_bboxes)

    if isinstance(merged_union, Polygon):
        merged_bboxes = [merged_union]
    elif isinstance(merged_union, MultiPolygon):
        merged_bboxes = list(merged_union.geoms)
    else:
        merged_bboxes = []

    filtered_bboxes = [poly for poly in merged_bboxes if poly.area >= minimal_bbox_area]

    return gpd.GeoDataFrame(geometry=filtered_bboxes, crs=gdf.crs)

def run_change_detection(old_path, new_path, buffer_dist, frac_buf_dist,
                          area_thresh, max_distance, num_bati, minimal_bbox_area,
                          output_prefix="change"):

    gdf_old = clean_gdf(gpd.read_file(old_path))
    gdf_new = clean_gdf(gpd.read_file(new_path))

    gdf_old_buf = gdf_old.buffer(buffer_dist, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
    gdf_new_buf = gdf_new.buffer(buffer_dist, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)

    gdf_old_buf = gpd.GeoDataFrame(geometry=gdf_old_buf, crs=gdf_old.crs)
    gdf_new_buf = gpd.GeoDataFrame(geometry=gdf_new_buf, crs=gdf_new.crs)

    buf_demolitions = gpd.overlay(gdf_old_buf, gdf_new_buf, how="difference")
    buf_constructions = gpd.overlay(gdf_new_buf, gdf_old_buf, how="difference")

    neg_buf = -(1 + frac_buf_dist) * buffer_dist
    demolitions = gpd.GeoDataFrame(geometry=buf_demolitions.buffer(
        neg_buf, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre), crs=gdf_old.crs)
    constructions = gpd.GeoDataFrame(geometry=buf_constructions.buffer(
        neg_buf, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre), crs=gdf_old.crs)

    demolitions.to_file(f"{output_prefix}_noisy_demolitions.geojson", driver="GeoJSON")
    constructions.to_file(f"{output_prefix}_noisy_constructions.geojson", driver="GeoJSON")

    demo_boxes = cluster_and_bbox_flexible_merged(demolitions, area_thresh, max_distance, num_bati, minimal_bbox_area)
    cons_boxes = cluster_and_bbox_flexible_merged(constructions, area_thresh, max_distance, num_bati, minimal_bbox_area)

    demo_boxes.to_file(f"{output_prefix}_demolition_boxes.geojson", driver="GeoJSON")
    cons_boxes.to_file(f"{output_prefix}_construction_boxes.geojson", driver="GeoJSON")

if __name__ == "__main__":
    # === USER PARAMETERS ===
    old_path = "/home/adelb/Downloads/SAT-COTIA_Detection_Line_PCRS_2021_avec_Attributs.geojson"
    new_path = "/home/adelb/Documents/Bpartners/GeoJson/spot-2024-z16/line_spot_2024_z16.geojson"

    buffer_dist = 10
    frac_buf_dist = 0.05

    area_thresh = 60.0
    max_distance = 100.0
    num_bati = 1
    minimal_bbox_area = 500.0

    run_change_detection(
        old_path, new_path,
        buffer_dist, frac_buf_dist,
        area_thresh, max_distance, num_bati, minimal_bbox_area,
        output_prefix="change"
    )
