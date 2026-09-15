"""
Spatial Analysis for Hotspot Forecasting
Implements Kernel Density Estimation (KDE) and Getis-Ord Gi* statistics
Report-based approach - analyzes reports directly without cluster dependency
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
import geojson
from geojson import Feature, FeatureCollection, Polygon, Point

from config import (
    KDE_BANDWIDTH,
    GRID_RESOLUTION,
    GI_STATISTIC_THRESHOLD,
    P_VALUE_THRESHOLD,
    RISK_SCORE_HIGH_THRESHOLD,
    RISK_SCORE_MEDIUM_THRESHOLD,
    GRID_CELL_SIZE_METERS,
    MIN_REPORTS_PER_HOTSPOT,
    MIN_REPORTS_FOR_HIGH_RISK,
    MIN_REPORTS_FOR_MEDIUM_RISK,
    FORECAST_LOOKBACK_WEEKS,
    FORECAST_MIN_CONSISTENCY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialAnalyzer:
    """Performs spatial analysis for hotspot detection"""
    
    def __init__(self):
        logger.info("SpatialAnalyzer initialized")
    
    def compute_kde(
        self,
        points: List[Tuple[float, float]],
        bounding_box: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute Kernel Density Estimation for a set of points
        
        Args:
            points: List of (longitude, latitude) tuples
            bounding_box: Optional dict with 'min_lon', 'max_lon', 'min_lat', 'max_lat'
        
        Returns:
            Dict with 'grid_x', 'grid_y', 'density' arrays
        """
        if len(points) < 2:
            logger.warning("Not enough points for KDE (need at least 2)")
            return None
        
        # Extract coordinates
        lons = np.array([p[0] for p in points])
        lats = np.array([p[1] for p in points])
        
        # Determine bounding box
        if bounding_box:
            min_lon, max_lon = bounding_box['min_lon'], bounding_box['max_lon']
            min_lat, max_lat = bounding_box['min_lat'], bounding_box['max_lat']
        else:
            min_lon, max_lon = lons.min(), lons.max()
            min_lat, max_lat = lats.min(), lats.max()
            # Add padding
            lon_pad = (max_lon - min_lon) * 0.1
            lat_pad = (max_lat - min_lat) * 0.1
            min_lon -= lon_pad
            max_lon += lon_pad
            min_lat -= lat_pad
            max_lat += lat_pad
        
        # Create grid
        grid_x = np.arange(min_lon, max_lon, GRID_RESOLUTION)
        grid_y = np.arange(min_lat, max_lat, GRID_RESOLUTION)
        xx, yy = np.meshgrid(grid_x, grid_y)
        
        # Compute KDE
        try:
            kde = gaussian_kde(np.vstack([lons, lats]), bw_method=KDE_BANDWIDTH)
            density = kde(np.vstack([xx.ravel(), yy.ravel()]))
            density = density.reshape(xx.shape)
            
            logger.info(f"KDE computed on {len(grid_x)}x{len(grid_y)} grid")
            
            return {
                'grid_x': grid_x,
                'grid_y': grid_y,
                'density': density,
                'min_lon': min_lon,
                'max_lon': max_lon,
                'min_lat': min_lat,
                'max_lat': max_lat
            }
            
        except Exception as e:
            logger.error(f"Error computing KDE: {e}")
            return None
    
    def compute_getis_ord_gi(
        self,
        points: List[Tuple[float, float]],
        values: Optional[List[float]] = None,
        spatial_weights: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute Getis-Ord Gi* statistic for hotspot detection
        
        Args:
            points: List of (longitude, latitude) tuples
            values: Optional list of values (e.g., report counts). If None, uses uniform weights
            spatial_weights: Optional precomputed spatial weights matrix
        
        Returns:
            Dict with 'gi_star', 'p_value', 'z_score' arrays
        """
        if len(points) < 3:
            logger.warning("Not enough points for Getis-Ord Gi* (need at least 3)")
            return None
        
        n = len(points)
        
        # Use uniform values if not provided
        if values is None:
            values = np.ones(n)
        else:
            values = np.array(values)
        
        # Compute spatial weights if not provided (inverse distance weighting)
        if spatial_weights is None:
            coords = np.array(points)
            dist_matrix = squareform(pdist(coords))
            # Avoid division by zero
            dist_matrix[dist_matrix == 0] = 1e-10
            spatial_weights = 1.0 / dist_matrix
            # Normalize row-wise
            spatial_weights = spatial_weights / spatial_weights.sum(axis=1, keepdims=True)
        
        # Compute Getis-Ord Gi*
        gi_star = np.zeros(n)
        p_values = np.zeros(n)
        z_scores = np.zeros(n)
        
        # Global statistics
        mean_y = np.mean(values)
        std_y = np.std(values)
        
        if std_y == 0:
            logger.warning("Zero variance in values, cannot compute Gi*")
            # Return default values when variance is zero
            return {
                'gi_star': np.zeros(n),
                'p_value': np.ones(n),
                'z_score': np.zeros(n),
                'significant': np.zeros(n, dtype=bool)
            }
        
        for i in range(n):
            # Local sum
            w_i = spatial_weights[i]
            sum_y_local = np.sum(w_i * values)
            sum_w = np.sum(w_i)
            
            # Expected value
            expected = mean_y * sum_w
            
            # Variance
            s_squared = np.sum((values - mean_y) ** 2) / n
            sum_w_squared = np.sum(w_i ** 2)
            variance = s_squared * ((n * sum_w_squared - sum_w ** 2) / (n - 1))
            
            # Standard deviation
            std_dev = np.sqrt(variance) if variance > 0 else 1e-10
            
            # Gi* statistic
            gi_star[i] = (sum_y_local - expected) / std_dev
            
            # Z-score (same as Gi* for this formulation)
            z_scores[i] = gi_star[i]
            
            # P-value (two-tailed)
            from scipy.stats import norm
            p_values[i] = 2 * (1 - norm.cdf(abs(gi_star[i])))
        
        logger.info(f"Getis-Ord Gi* computed for {n} points")
        
        return {
            'gi_star': gi_star,
            'p_value': p_values,
            'z_score': z_scores,
            'significant': (p_values < P_VALUE_THRESHOLD) & (np.abs(gi_star) > GI_STATISTIC_THRESHOLD)
        }
    
    def analyze_report_hotspots(
        self,
        reports: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Analyze reports to identify hotspot regions using spatial binning
        
        Args:
            reports: List of report dictionaries with location data
        
        Returns:
            Dict mapping region_id to hotspot analysis results
        """
        if not reports:
            logger.warning("No reports provided for hotspot analysis")
            return {}
        
        # Extract points
        points = [(r['longitude'], r['latitude']) for r in reports if 'longitude' in r and 'latitude' in r]
        
        if len(points) < MIN_REPORTS_PER_HOTSPOT:
            logger.warning(f"Not enough reports for hotspot analysis (need at least {MIN_REPORTS_PER_HOTSPOT})")
            return {}
        
        logger.info(f"Analyzing {len(reports)} reports for hotspots")
        
        # Perform spatial binning to create regions
        region_analyses = self._bin_reports_to_regions(reports)
        
        # Analyze each region for hotspot potential
        hotspot_results = {}
        for region_id, region_data in region_analyses.items():
            if region_data['report_count'] >= MIN_REPORTS_PER_HOTSPOT:
                analysis = self._analyze_region_hotspot(region_id, region_data)
                hotspot_results[region_id] = analysis
        
        logger.info(f"Identified {len(hotspot_results)} hotspot regions")
        return hotspot_results
    
    def _bin_reports_to_regions(self, reports: List[Dict]) -> Dict[str, Dict]:
        """
        Bin reports into spatial grid cells
        
        Args:
            reports: List of report dictionaries
        
        Returns:
            Dict mapping region_id to region data
        """
        regions = {}
        
        # Convert grid cell size from meters to degrees (approximate)
        # 1 degree ≈ 111km at equator
        cell_size_deg = GRID_CELL_SIZE_METERS / 111000.0
        
        for report in reports:
            lon = report.get('longitude')
            lat = report.get('latitude')
            
            if lon is None or lat is None:
                continue
            
            # Calculate grid cell indices
            grid_x = int(lon / cell_size_deg)
            grid_y = int(lat / cell_size_deg)
            region_id = f"{grid_x}_{grid_y}"
            
            if region_id not in regions:
                regions[region_id] = {
                    'region_id': region_id,
                    'reports': [],
                    'center_lon': 0,
                    'center_lat': 0
                }
            
            regions[region_id]['reports'].append(report)
            regions[region_id]['center_lon'] += lon
            regions[region_id]['center_lat'] += lat
        
        # Calculate region centers
        for region_id, region_data in regions.items():
            count = len(region_data['reports'])
            region_data['center_lon'] /= count
            region_data['center_lat'] /= count
            region_data['report_count'] = count
        
        return regions
    
    def _analyze_region_hotspot(self, region_id: str, region_data: Dict) -> Dict:
        """
        Analyze a single region for hotspot potential using density-based approach
        
        Args:
            region_id: Unique identifier for the region
            region_data: Region data including reports and center coordinates
        
        Returns:
            Dict with hotspot analysis results
        """
        reports = region_data['reports']
        report_count = len(reports)
        
        # Extract points for spatial reference (kept for potential future use)
        points = [(r['longitude'], r['latitude']) for r in reports if 'longitude' in r and 'latitude' in r]
        
        # Skip regions with too few reports
        if report_count < MIN_REPORTS_PER_HOTSPOT:
            return {
                'region_id': region_id,
                'risk_score': 0.0,
                'risk_level': 'low',
                'is_hotspot': False,
                'report_count': report_count,
                'gi_star': 0.0,
                'p_value': 1.0,
                'is_significant': False,
                'region_center_lat': region_data['center_lat'],
                'region_center_lng': region_data['center_lon']
            }
        
        # Density-based hotspot detection (3+ reports = high risk, 2 reports = medium risk)
        if report_count >= MIN_REPORTS_FOR_HIGH_RISK:
            risk_level = 'high'
            risk_score = min(1.0, (report_count - MIN_REPORTS_FOR_HIGH_RISK + 1) / 10.0 + 0.7)  # Normalize to 0.7-1.0 range
            is_hotspot = True
        elif report_count >= MIN_REPORTS_FOR_MEDIUM_RISK:
            risk_level = 'medium'
            risk_score = min(0.6, (report_count - MIN_REPORTS_FOR_MEDIUM_RISK + 1) / 10.0 + 0.3)  # Normalize to 0.3-0.6 range
            is_hotspot = True
        else:
            risk_level = 'low'
            risk_score = 0.0
            is_hotspot = False
        
        return {
            'region_id': region_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'is_hotspot': is_hotspot,
            'report_count': report_count,
            'gi_star': 0.0,  # Kept for backward compatibility
            'p_value': 1.0,  # Kept for backward compatibility
            'is_significant': is_hotspot,  # Hotspots are considered significant
            'region_center_lat': region_data['center_lat'],
            'region_center_lng': region_data['center_lon'],
            'region_radius_meters': GRID_CELL_SIZE_METERS
        }
    
    def forecast_hotspots_persistence(
        self,
        historical_reports: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Forecast future hotspots based on spatial persistence patterns
        
        Args:
            historical_reports: Dict mapping week identifiers to report lists
        
        Returns:
            Dict mapping region_id to forecast analysis with confidence scores
        """
        logger.info(f"Analyzing persistence patterns across {len(historical_reports)} weeks")
        
        # Analyze each week to get hotspot regions
        weekly_hotspots = {}
        for week_key, reports in historical_reports.items():
            if not reports:
                weekly_hotspots[week_key] = {}
                continue
            
            # Perform spatial binning for this week
            regions = self._bin_reports_to_regions(reports)
            
            # Analyze each region for hotspot status
            week_hotspots = {}
            for region_id, region_data in regions.items():
                report_count = len(region_data['reports'])
                
                # Determine if this was a hotspot
                if report_count >= MIN_REPORTS_FOR_HIGH_RISK:
                    risk_level = 'high'
                    is_hotspot = True
                elif report_count >= MIN_REPORTS_FOR_MEDIUM_RISK:
                    risk_level = 'medium'
                    is_hotspot = True
                else:
                    risk_level = 'low'
                    is_hotspot = False
                
                if is_hotspot:
                    week_hotspots[region_id] = {
                        'risk_level': risk_level,
                        'report_count': report_count,
                        'center_lat': region_data['center_lat'],
                        'center_lng': region_data['center_lon']
                    }
            
            weekly_hotspots[week_key] = week_hotspots
            logger.info(f"Week {week_key}: {len(week_hotspots)} hotspot regions")
        
        # Analyze persistence patterns
        forecast_results = {}
        region_weekly_status = {}
        
        # Build matrix of which regions were hotspots in which weeks
        for week_key, hotspots in weekly_hotspots.items():
            for region_id, hotspot_data in hotspots.items():
                if region_id not in region_weekly_status:
                    region_weekly_status[region_id] = {
                        'total_weeks': 0,
                        'hotspot_weeks': 0,
                        'risk_levels': [],
                        'report_counts': [],
                        'center_lat': hotspot_data['center_lat'],
                        'center_lng': hotspot_data['center_lng']
                    }
                
                region_weekly_status[region_id]['total_weeks'] += 1
                region_weekly_status[region_id]['hotspot_weeks'] += 1
                region_weekly_status[region_id]['risk_levels'].append(hotspot_data['risk_level'])
                region_weekly_status[region_id]['report_counts'].append(hotspot_data['report_count'])
        
        # Generate forecasts based on persistence
        for region_id, status in region_weekly_status.items():
            hotspot_weeks = status['hotspot_weeks']
            total_weeks = status['total_weeks']
            
            # Calculate persistence score
            persistence_ratio = hotspot_weeks / total_weeks if total_weeks > 0 else 0
            
            # Determine forecast confidence
            if hotspot_weeks >= FORECAST_MIN_CONSISTENCY:
                confidence = min(1.0, persistence_ratio + 0.3)  # Boost confidence for consistent hotspots
                is_forecasted = True
            else:
                confidence = persistence_ratio
                is_forecasted = False
            
            # Determine predicted risk level
            if status['risk_levels']:
                most_common_risk = max(set(status['risk_levels']), key=status['risk_levels'].count)
                avg_report_count = sum(status['report_counts']) / len(status['report_counts'])
            else:
                most_common_risk = 'low'
                avg_report_count = 0
            
            if is_forecasted:
                forecast_results[region_id] = {
                    'region_id': region_id,
                    'is_forecasted': True,
                    'confidence': confidence,
                    'predicted_risk_level': most_common_risk,
                    'predicted_report_count': avg_report_count,
                    'persistence_weeks': hotspot_weeks,
                    'total_weeks_analyzed': total_weeks,
                    'region_center_lat': status['center_lat'],
                    'region_center_lng': status['center_lng'],
                    'forecast_type': 'persistence'
                }
        
        logger.info(f"Forecasted {len(forecast_results)} persistent hotspots")
        return forecast_results
    
    def generate_heatmap_geojson(
        self,
        reports: List[Dict]
    ) -> FeatureCollection:
        """
        Generate heatmap GeoJSON using Kernel Density Estimation
        
        Args:
            reports: List of reports with longitude/latitude
        
        Returns:
            GeoJSON FeatureCollection with heatmap point features (intensity values)
        """
        # Extract coordinates
        coords = []
        for report in reports:
            if 'longitude' in report and 'latitude' in report:
                coords.append([report['longitude'], report['latitude']])
        
        if len(coords) < 2:
            return FeatureCollection([])
        
        coords = np.array(coords)
        
        # Perform KDE
        try:
            kde = gaussian_kde(coords.T, bw_method=KDE_BANDWIDTH)
        except:
            # Fallback if KDE fails (e.g., all points at same location)
            return FeatureCollection([])
        
        # Generate grid for heatmap
        lons = coords[:, 0]
        lats = coords[:, 1]
        
        lon_min, lon_max = lons.min() - 0.01, lons.max() + 0.01
        lat_min, lat_max = lats.min() - 0.01, lats.max() + 0.01
        
        # Create grid (100x100 resolution)
        grid_lons = np.linspace(lon_min, lon_max, 100)
        grid_lats = np.linspace(lat_min, lat_max, 100)
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
        
        # Evaluate KDE on grid
        grid_coords = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
        density = kde(grid_coords).reshape(100, 100)
        
        # Normalize density to 0-1 range
        if density.max() > 0:
            density = density / density.max()
        
        # Generate GeoJSON features (points with intensity)
        features = []
        for i in range(100):
            for j in range(100):
                intensity = density[i, j]
                if intensity > 0.05:  # Only include points with significant intensity
                    feature = Feature(
                        geometry=Point([grid_lons[j], grid_lats[i]]),
                        properties={
                            'intensity': float(intensity),
                            'type': 'heatmap'
                        }
                    )
                    features.append(feature)
        
        return FeatureCollection(features)
    
    def generate_hotspot_geojson(
        self,
        hotspot_analyses: Dict[str, Dict]
    ) -> FeatureCollection:
        """
        Generate GeoJSON FeatureCollection for hotspot visualization
        
        Args:
            hotspot_analyses: Dict mapping region_id to analysis results
        
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for region_id, analysis in hotspot_analyses.items():
            if not analysis.get('is_hotspot', False):
                continue
            
            center_lat = analysis.get('region_center_lat')
            center_lng = analysis.get('region_center_lng')
            radius_meters = analysis.get('region_radius_meters', GRID_CELL_SIZE_METERS)
            
            if center_lat is None or center_lng is None:
                continue
            
            # Convert radius to degrees (approximate)
            radius_deg = radius_meters / 111000.0
            
            # Create a circular buffer polygon (simplified as square)
            polygon_coords = [
                [
                    [center_lng - radius_deg, center_lat - radius_deg],
                    [center_lng + radius_deg, center_lat - radius_deg],
                    [center_lng + radius_deg, center_lat + radius_deg],
                    [center_lng - radius_deg, center_lat + radius_deg],
                    [center_lng - radius_deg, center_lat - radius_deg]
                ]
            ]
            
            # Determine color based on risk level
            risk_level = analysis.get('risk_level', 'low')
            if risk_level == 'high':
                fill_color = '#ff0000'  # Red
            elif risk_level == 'medium':
                fill_color = '#ffff00'  # Yellow
            else:
                fill_color = '#00ff00'  # Green
            
            feature = Feature(
                geometry=Polygon(polygon_coords),
                properties={
                    'region_id': region_id,
                    'risk_score': analysis.get('risk_score', 0),
                    'risk_level': risk_level,
                    'report_count': analysis.get('report_count', 0),
                    'gi_star': analysis.get('gi_star', 0),
                    'p_value': analysis.get('p_value', 1),
                    'center_lat': center_lat,
                    'center_lng': center_lng,
                    'fillColor': fill_color,
                    'fillOpacity': 0.5
                }
            )
            features.append(feature)
        
        return FeatureCollection(features)


# Singleton instance
_analyzer_instance = None

def get_spatial_analyzer() -> SpatialAnalyzer:
    """Get or create singleton SpatialAnalyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SpatialAnalyzer()
    return _analyzer_instance
