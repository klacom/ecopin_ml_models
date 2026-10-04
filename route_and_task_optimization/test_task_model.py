import json
from datetime import datetime, timedelta, timezone
from task_generation_model import TaskGenerationModel

def run_test():
    # Initialize the model
    # aging_factor=1.5 means each day adds 1.5 to priority
    model = TaskGenerationModel(aging_factor=1.5, max_detour_minutes=5.0, average_speed_kmh=30.0)

    now = datetime.now(timezone.utc)

    # Dummy Clusters
    clusters = [
        {
            'id': 'cluster_1',
            'base_severity': 5.0, # High base severity
            'created_at': now - timedelta(days=1), # Very recent
            'lat': 40.7128, 
            'lon': -74.0060 # NYC
        }
    ]

    # Dummy Individual Reports
    reports = [
        {
            'id': 'report_1_starving',
            'base_severity': 2.0, # Low base severity
            'created_at': now - timedelta(days=14), # 14 days old (aging: 14 * 1.5 = 21 priority points)
            'lat': 40.7300,
            'lon': -73.9900 
        },
        {
            'id': 'report_2_on_the_way',
            'base_severity': 1.0, # Very low severity
            'created_at': now - timedelta(days=1), # Very recent
            'lat': 40.7100,
            'lon': -74.0000 # Very close to the route to cluster_1
        },
        {
            'id': 'report_3_far_away',
            'base_severity': 1.0, 
            'created_at': now - timedelta(days=1),
            'lat': 40.8500, # Far away
            'lon': -73.9000 
        }
    ]

    # Officer Start Location
    start_location = {'lat': 40.7000, 'lon': -74.0100}

    print("--- Running Task Generation ---")
    tasks = model.generate_tasks(clusters, reports, start_location)
    
    print("\nGenerated Tasks Output:")
    print(json.dumps(tasks, indent=2))

if __name__ == "__main__":
    run_test()
