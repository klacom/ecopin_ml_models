from data_fetcher import fetch_reports
from spatial_analysis import run_dbscan_hotspots
from config import DEFAULT_BBOX

reports = fetch_reports('monthly', DEFAULT_BBOX)
print("Reports fetched:", len(reports))

result = run_dbscan_hotspots(reports)
print("Hotspot clusters:", result["hotspot_count"])
print("Heatmap points:", len(result["heatmap_geojson"]["features"]))
print("GeoJSON features:", len(result["geojson"]["features"]))

for rid, r in result["region_analyses"].items():
    print("  Cluster #" + str(r["rank"]) + ": " + str(r["report_count"]) + " reports, "
          + r["risk_level"] + " risk, top=" + str(r["top_issue_type"])
          + ", radius=" + str(r["region_radius_meters"]) + "m")
