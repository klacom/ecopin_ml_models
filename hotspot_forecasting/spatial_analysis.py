import numpy as np
import math
import uuid
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from config import (
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)

# Earth radius in meters for haversine metric
EARTH_RADIUS_M = 6371000.0


def meters_to_radians(meters):
    """Convert a distance in meters to radians for use with haversine metric."""
    return meters / EARTH_RADIUS_M


def compute_convex_hull_polygon(lats, lons):
    """
    Given arrays of latitudes and longitudes, return a GeoJSON-style polygon
    representing the convex hull. Returns None if hull cannot be formed.
    """
    coords = np.column_stack([lons, lats])

    if len(coords) < 3:
        # For 1–2 points, fall back to a small circle
        return None

    try:
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices].tolist()
        hull_pts.append(hull_pts[0])  # Close the polygon
        return hull_pts
    except Exception:
        # Degenerate case (collinear points)
        return None


def generate_circle_polygon(center_lon, center_lat, radius_meters, num_points=32):
    """Generate a smooth circular polygon around a center point."""
    polygon = []
    lat_degree_m = 111320.0
    lon_degree_m = 111320.0 * math.cos(math.radians(center_lat))

    for i in range(num_points):
        angle = math.radians(float(i) / num_points * 360.0)
        d_lon = (radius_meters * math.cos(angle)) / lon_degree_m
        d_lat = (radius_meters * math.sin(angle)) / lat_degree_m
        polygon.append([center_lon + d_lon, center_lat + d_lat])

    polygon.append(polygon[0])  # Close
    return polygon


def run_dbscan_hotspots(reports, bbox=None):
    """
    Main hotspot analysis function using DBSCAN clustering.

    Returns:
        dict with:
            - heatmap_geojson: all valid report points for smooth Leaflet heatmap
            - geojson: DBSCAN cluster polygons (convex hulls) as hotspot regions
            - region_analyses: dict of cluster metadata keyed by uuid
            - hotspot_count, total_reports, prediction_date, time_horizon
    """
    empty = {
        "region_analyses": {},
        "heatmap_geojson": {"type": "FeatureCollection", "features": []},
        "geojson": {"type": "FeatureCollection", "features": []},
        "hotspot_count": 0,
        "total_regions": 0,
        "total_reports": 0,
        "prediction_date": datetime.utcnow().isoformat(),
        "time_horizon": "weekly",
    }

    if not reports:
        print("[DBSCAN] No reports to process.")
        return empty

    # ── 1. Raw heatmap layer ──────────────────────────────────────────────────
    heatmap_features = []
    for r in reports:
        heatmap_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["longitude"], r["latitude"]],
            },
            "properties": {"intensity": 1.0},
        })

    heatmap_geojson = {"type": "FeatureCollection", "features": heatmap_features}

    # ── 2. DBSCAN clustering ──────────────────────────────────────────────────
    lats = np.array([r["latitude"] for r in reports])
    lons = np.array([r["longitude"] for r in reports])

    # DBSCAN with haversine requires coordinates in radians [lat, lon]
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
    eps_rad = meters_to_radians(DBSCAN_EPS_METERS)

    db = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)

    labels = db.labels_
    unique_labels = set(labels)
    noise_count = int(np.sum(labels == -1))
    cluster_count = len(unique_labels - {-1})

    print(f"[DBSCAN] Found {cluster_count} clusters, {noise_count} noise points "
          f"(eps={DBSCAN_EPS_METERS}m, min_samples={DBSCAN_MIN_SAMPLES})")

    # ── 3. Build cluster regions ──────────────────────────────────────────────
    region_analyses = {}
    geojson_features = []

    # Sort clusters by size descending so cluster #1 is the largest
    cluster_ids = sorted(
        [lbl for lbl in unique_labels if lbl != -1],
        key=lambda lbl: int(np.sum(labels == lbl)),
        reverse=True,
    )

    for rank, cluster_label in enumerate(cluster_ids, start=1):
        indices = np.where(labels == cluster_label)[0]
        report_count = len(indices)

        cluster_lats = lats[indices]
        cluster_lons = lons[indices]
        centroid_lat = float(np.mean(cluster_lats))
        centroid_lon = float(np.mean(cluster_lons))

        # Issue type breakdown
        issue_types = [reports[i].get("issue_type", "unknown") for i in indices]
        issue_counts = dict(Counter(issue_types))
        top_issue = max(issue_counts, key=issue_counts.get) if issue_counts else "unknown"

        # Convex hull polygon
        polygon_coords = compute_convex_hull_polygon(cluster_lats, cluster_lons)

        # Compute actual cluster radius (max distance from centroid to any point)
        lat_scale = 111320.0
        lon_scale = 111320.0 * math.cos(math.radians(centroid_lat))
        max_dist_m = 0.0
        for i in indices:
            dy = (lats[i] - centroid_lat) * lat_scale
            dx = (lons[i] - centroid_lon) * lon_scale
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_dist_m:
                max_dist_m = dist

        radius_meters = max(50.0, max_dist_m + 20.0)  # min 50m, +20m padding

        # Fall back to circle if hull failed (e.g., collinear points)
        if polygon_coords is None:
            polygon_coords = generate_circle_polygon(
                centroid_lon, centroid_lat, radius_meters
            )

        # Risk classification
        if report_count >= RISK_HIGH_THRESHOLD:
            risk_level = "high"
        elif report_count >= RISK_MEDIUM_THRESHOLD:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Risk score 0–1 based on report count (soft cap at 20)
        risk_score = min(1.0, report_count / 20.0)

        region_id = str(uuid.uuid4())

        region_analyses[region_id] = {
            "rank": rank,
            "risk_score": risk_score,
            "report_count": report_count,
            "risk_level": risk_level,
            "region_center_lat": centroid_lat,
            "region_center_lng": centroid_lon,
            "region_radius_meters": round(radius_meters, 1),
            "top_issue_type": top_issue,
            "issue_breakdown": issue_counts,
            "is_hotspot": True,
            # Legacy fields kept for DB compatibility
            "gi_star": risk_score * 5.0,
            "p_value": 0.01,
            "is_significant": True,
        }

        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords],
            },
            "properties": {
                "region_id": region_id,
                "rank": rank,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "report_count": report_count,
                "top_issue_type": top_issue,
                "issue_breakdown": issue_counts,
                "center_lat": centroid_lat,
                "center_lng": centroid_lon,
            },
        })

    geojson_out = {"type": "FeatureCollection", "features": geojson_features}

    print(f"[DBSCAN] Generated {len(region_analyses)} hotspot regions")

    return {
        "region_analyses": region_analyses,
        "heatmap_geojson": heatmap_geojson,
        "geojson": geojson_out,
        "hotspot_count": len(region_analyses),
        "total_regions": len(region_analyses),
        "total_reports": len(reports),
        "noise_count": noise_count,
        "prediction_date": datetime.utcnow().isoformat(),
        "time_horizon": "weekly",  # overridden by caller
        "algorithm": "DBSCAN",
        "params": {
            "eps_meters": DBSCAN_EPS_METERS,
            "min_samples": DBSCAN_MIN_SAMPLES,
        },
    }
