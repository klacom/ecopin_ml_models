import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class TaskGenerationModel:
    def __init__(
        self, 
        aging_factor: float = 1.5, 
        max_escalation_cap: float = 10.0,
        max_detour_minutes: float = 5.0, 
        max_detour_time_per_task: float = 15.0,
        average_speed_kmh: float = 30.0
    ):
        """
        Model to handle automatic generation of tasks with Aging Priority and On-the-Way Route Optimization.
        
        :param aging_factor: The multiplier for how much priority increases per day unresolved.
        :param max_escalation_cap: The maximum priority points that can be added via aging.
        :param max_detour_minutes: Maximum allowed detour time in minutes to pick up an isolated report.
        :param max_detour_time_per_task: Maximum cumulative detour time allowed per primary task.
        :param average_speed_kmh: Assumed average speed in km/h for detour calculations.
        """
        self.aging_factor = aging_factor
        self.max_escalation_cap = max_escalation_cap
        self.max_detour_minutes = max_detour_minutes
        self.max_detour_time_per_task = max_detour_time_per_task
        self.average_speed_kmh = average_speed_kmh

    def _calculate_days_unresolved(self, created_at: datetime) -> int:
        delta = datetime.now(timezone.utc) - created_at
        return max(0, delta.days)

    def calculate_priority_score(self, base_severity: float, created_at: datetime) -> float:
        """
        PriorityScore = BaseSeverity + min((DaysUnresolved * AgingFactor), MaxEscalationCap)
        """
        days_unresolved = self._calculate_days_unresolved(created_at)
        escalation = min(days_unresolved * self.aging_factor, self.max_escalation_cap)
        return base_severity + escalation

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance in kilometers between two points on the earth.
        """
        R = 6371.0 # Radius of earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def _calculate_detour_time_minutes(self, start: Dict, end: Dict, waypoint: Dict) -> float:
        """
        Estimates the detour time added by visiting the waypoint between start and end.
        start, end, and waypoint are dicts with 'lat' and 'lon'.
        """
        dist_direct = self._haversine_distance(start['lat'], start['lon'], end['lat'], end['lon'])
        dist_via_waypoint = (
            self._haversine_distance(start['lat'], start['lon'], waypoint['lat'], waypoint['lon']) +
            self._haversine_distance(waypoint['lat'], waypoint['lon'], end['lat'], end['lon'])
        )
        extra_distance_km = dist_via_waypoint - dist_direct
        # time = distance / speed
        extra_time_hours = extra_distance_km / self.average_speed_kmh
        return extra_time_hours * 60.0

    def generate_tasks(
        self, 
        clusters: List[Dict[str, Any]], 
        individual_reports: List[Dict[str, Any]],
        start_location: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generates tasks by selecting high priority targets (clusters or individual reports) 
        and attempting to add "on-the-way" isolated reports that don't add too much detour time.
        
        clusters: List of cluster dicts (e.g. id, lat, lon, base_severity, created_at)
        individual_reports: List of report dicts (e.g. id, lat, lon, base_severity, created_at)
        start_location: Dict with 'lat', 'lon' for the officer's start point
        """
        
        # 1. Age-Based Priority Escalation (Aging)
        # Calculate PriorityScore for all targets
        all_targets = []
        
        for cluster in clusters:
            cluster['priority_score'] = self.calculate_priority_score(
                cluster['base_severity'], cluster['created_at']
            )
            cluster['is_cluster'] = True
            all_targets.append(cluster)
            
        for report in individual_reports:
            report['priority_score'] = self.calculate_priority_score(
                report['base_severity'], report['created_at']
            )
            report['is_cluster'] = False
            report['assigned'] = False
            all_targets.append(report)

        # Sort all potential primary targets by priority score (descending)
        # This solves starvation: an old individual report will outrank a new cluster
        targets_sorted = sorted(all_targets, key=lambda x: x['priority_score'], reverse=True)
        
        tasks = []
        
        # 2. "On-the-Way" Route Optimization (Detour Tolerance)
        for target in targets_sorted:
            # If target is a report and it was already picked up on the way to a higher priority target, skip it.
            if not target['is_cluster'] and target.get('assigned', False):
                continue
                
            task = {
                'primary_target_id': target['id'],
                'primary_target_is_cluster': target['is_cluster'],
                'primary_priority': target['priority_score'],
                'target_lat': target['lat'],
                'target_lon': target['lon'],
                'included_reports': [],
                'cumulative_detour_minutes': 0.0
            }
            
            # If the primary target is an individual report, mark it as assigned
            if not target['is_cluster']:
                target['assigned'] = True
            
            # Check for isolated reports along the route from start_location to this target
            for report in individual_reports:
                if not report['assigned']:
                    # Calculate detour time
                    detour_mins = self._calculate_detour_time_minutes(
                        start=start_location, 
                        end={'lat': target['lat'], 'lon': target['lon']}, 
                        waypoint={'lat': report['lat'], 'lon': report['lon']}
                    )
                    
                    if detour_mins <= self.max_detour_minutes:
                        if task['cumulative_detour_minutes'] + detour_mins <= self.max_detour_time_per_task:
                            task['included_reports'].append({
                                'report_id': report['id'],
                                'detour_time_minutes': detour_mins,
                                'priority_score': report['priority_score']
                            })
                            task['cumulative_detour_minutes'] += detour_mins
                            report['assigned'] = True
                        
            tasks.append(task)

        return tasks
