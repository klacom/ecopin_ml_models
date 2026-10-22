import json
import time
import unittest
from datetime import datetime, timedelta, timezone

from zone_forecaster import ZoneTimeForecaster, build_projection, regularize_history


NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
MONDAY = datetime(2026, 10, 12, tzinfo=timezone.utc)
ZONE = {
    'id': 'stable-zone', 'zone_name': 'Test zone', 'zone_type': 'h3',
    'centroid_lat': 14.57, 'centroid_lng': 121.08,
    'boundary_geojson': {'type': 'Polygon', 'coordinates': [[[121, 14], [122, 14], [122, 15], [121, 14]]]},
}


class ZoneForecastTests(unittest.TestCase):
    def test_regularization_zero_fills_and_excludes_current_incomplete_week(self):
        rows = [
            {'week_start': (MONDAY - timedelta(weeks=2)).isoformat(), 'report_count': 4},
            {'week_start': MONDAY.isoformat(), 'report_count': 999},
        ]
        points = regularize_history(rows, NOW)
        self.assertEqual(len(points), 52)
        self.assertEqual(points[-1]['actual_count'], 0)
        self.assertEqual(points[-2]['actual_count'], 4)
        self.assertEqual(points[-1]['week_start'], '2026-10-05')

    def test_exact_partial_weeks_and_comparable_baseline(self):
        for days in (7, 30, 90):
            result = ZoneTimeForecaster(days).forecast_zone([7] * 52, NOW)
            self.assertAlmostEqual(result['predicted_count'], days, places=2)
            self.assertAlmostEqual(result['growth_velocity_percent'], 0, places=1)
            self.assertAlmostEqual(sum(p['predicted_count'] for p in result['forecast_points']), days, places=1)
            self.assertLessEqual(result['predicted_lower_bound'], result['predicted_count'])
            self.assertGreater(result['predicted_upper_bound'], result['predicted_count'])

    def test_trend_horizons_share_monotonic_nonnegative_trajectory(self):
        values = [i // 3 + 1 for i in range(52)]
        results = [ZoneTimeForecaster(d).forecast_zone(values, NOW) for d in (7, 30, 90)]
        self.assertGreater(results[0]['predicted_count'], values[-1])
        self.assertLess(results[0]['predicted_count'], results[1]['predicted_count'])
        self.assertLess(results[1]['predicted_count'], results[2]['predicted_count'])
        declining = ZoneTimeForecaster(90).forecast_zone(list(range(51, -1, -1)), NOW)
        self.assertGreaterEqual(declining['predicted_lower_bound'], 0)
        self.assertGreaterEqual(declining['predicted_count'], 0)

    def test_sparse_and_empty_history_have_finite_statistical_intervals(self):
        for values in ([], [0] * 52, [0] * 51 + [1], [1, 2]):
            result = ZoneTimeForecaster(30).forecast_zone(values, NOW)
            json.dumps(result, allow_nan=False)
            self.assertLessEqual(result['predicted_lower_bound'], result['predicted_count'])
            self.assertGreaterEqual(result['predicted_upper_bound'], result['predicted_count'])
            self.assertGreaterEqual(result['predicted_upper_bound'], 0)

    def test_risk_follows_prose_thresholds(self):
        classify = ZoneTimeForecaster.classify_risk
        self.assertEqual(classify(15, 40), 'medium')
        self.assertEqual(classify(15.01, 0), 'high')
        self.assertEqual(classify(1, 40.01), 'high')
        self.assertEqual(classify(4.99, 0), 'low')

    def test_projection_preserves_zero_zones_and_forward_dates(self):
        response = build_projection([ZONE], [], '30d', NOW)
        self.assertEqual(response['model_version'], 'zone-timeseries-v1')
        self.assertEqual(response['forecast_date_start'], NOW.isoformat())
        self.assertEqual(response['forecast_date_end'], (NOW + timedelta(days=30)).isoformat())
        self.assertEqual(response['total_regions'], 1)
        self.assertEqual(response['region_analyses']['stable-zone']['predicted_count'], 0)
        self.assertEqual(response['geojson']['features'][0]['geometry'], ZONE['boundary_geojson'])
        self.assertTrue(0 <= response['region_analyses']['stable-zone']['risk_score'] <= 1)

    def test_thirty_zone_inference_under_one_point_five_seconds(self):
        zones = [{**ZONE, 'id': f'z-{i}'} for i in range(30)]
        rows = [{'zone_id': z['id'], 'week_start': (MONDAY - timedelta(weeks=52-i)).isoformat(),
                 'report_count': 3 + i // 5 + i % 3} for z in zones for i in range(52)]
        started = time.perf_counter()
        build_projection(zones, rows, '90d', NOW)
        self.assertLess(time.perf_counter() - started, 1.5)


if __name__ == '__main__':
    unittest.main()
