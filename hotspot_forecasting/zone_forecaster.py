"""Weekly count forecasts on stable zones; UTC weeks start on Monday.

Holt intervals use an innovation covariance approximation with a Poisson variance
floor. Sparse zones use Gamma-Poisson predictive quantiles. Neither is a calibrated
hotspot probability, and weekly counts assume uniform incidence within each week.
"""
from datetime import datetime, timedelta, timezone
import math
import warnings

import numpy as np
from scipy.stats import nbinom
from statsmodels.tsa.holtwinters import ExponentialSmoothing


HORIZONS = {'7d': 7, '30d': 30, '90d': 90}
MODEL_VERSION = 'zone-timeseries-v1'
HISTORY_WEEKS = 52


def utc_datetime(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def week_start(value):
    value = utc_datetime(value)
    return value.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=value.weekday())


def regularize_history(rows, now=None):
    """Keep 52 complete weeks, including weeks with no eligible reports."""
    end = week_start(now)
    counts = {}
    for row in rows:
        start = week_start(row['week_start'])
        count = float(row['report_count'])
        if not math.isfinite(count) or count < 0:
            raise ValueError('History counts must be finite and nonnegative')
        counts[start] = counts.get(start, 0) + count
    return [{'week_start': (end - timedelta(weeks=i)).date().isoformat(),
             'actual_count': counts.get(end - timedelta(weeks=i), 0)}
            for i in range(HISTORY_WEEKS, 0, -1)]


def overlap_fraction(start, end, bucket_start):
    overlap = min(end, bucket_start + timedelta(days=7)) - max(start, bucket_start)
    return max(0, overlap.total_seconds() / (7 * 86400))


class ZoneTimeForecaster:
    def __init__(self, horizon_days=30):
        if horizon_days not in HORIZONS.values():
            raise ValueError('Unsupported forecast horizon')
        self.horizon_days = horizon_days

    @staticmethod
    def classify_risk(predicted_count, velocity=0):
        # The prose is authoritative: high is >15 reports OR >40% surge.
        if predicted_count > 15 or velocity > 40:
            return 'high'
        return 'medium' if predicted_count >= 5 else 'low'

    def forecast_zone(self, historical_weekly_counts, now=None):
        now = utc_datetime(now)
        origin = week_start(now)
        end = now + timedelta(days=self.horizon_days)
        data = np.asarray(historical_weekly_counts, dtype=float)
        if np.any(~np.isfinite(data)) or np.any(data < 0):
            raise ValueError('History counts must be finite and nonnegative')
        # Always fit the same trajectory regardless of the requested horizon.
        steps = 14
        sparse = len(data) < 4 or np.count_nonzero(data) < 4
        model = None
        if not sparse and np.ptp(data) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    model = ExponentialSmoothing(data, trend='add', damped_trend=True,
                                                 initialization_method='estimated').fit(use_brute=False)
                means = np.maximum(0, np.asarray(model.forecast(steps)))
                if not np.all(np.isfinite(means)) or not model.mle_retvals.success:
                    raise ValueError('Forecast optimization did not converge')
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                model = None
                sparse = True
        if model is None:
            means = np.full(steps, float(np.mean(data if sparse else data[-4:])) if len(data) else 0)

        fractions = np.array([overlap_fraction(now, end, origin + timedelta(weeks=i)) for i in range(steps)])
        total = float(fractions @ means)
        if sparse:
            # Jeffreys prior prevents fabricated certainty for empty zones.
            exposure = max(1, len(data))
            shape = float(np.sum(data)) + .5

            def interval(weights):
                future_exposure = float(np.sum(weights))
                if future_exposure == 0:
                    return 0.0, 0.0
                probability = exposure / (exposure + future_exposure)
                return float(nbinom.ppf(.025, shape, probability)), float(nbinom.ppf(.975, shape, probability))
        else:
            alpha = float(model.params['smoothing_level']) if model is not None else 0
            beta = float(model.params['smoothing_trend']) if model is not None else 0
            phi = float(model.params['damping_trend']) if model is not None else .98
            innovations = np.eye(steps)
            for h in range(steps):
                for j in range(h):
                    innovations[h, j] = alpha * (1 + beta * sum(phi ** k for k in range(1, h - j + 1)))
            residual_variance = float(np.mean(np.square(model.resid))) if model is not None else 0
            variance = max(residual_variance, float(np.mean(data)), .01)
            covariance = variance * innovations @ innovations.T

            def interval(weights):
                mean = float(weights @ means)
                # Poisson thinning adds within-week uncertainty for partial weeks.
                var = float(weights @ covariance @ weights + np.sum(weights * (1 - weights) * means))
                margin = 1.96 * math.sqrt(max(0, var))
                return max(0, mean - margin), mean + margin

        lower, upper = interval(fractions)
        points = []
        for i, weight in enumerate(fractions):
            if weight <= 0:
                continue
            selected = np.zeros(steps)
            selected[i] = weight
            point = float(weight * means[i])
            low, high = interval(selected)
            points.append({'week_start': (origin + timedelta(weeks=i)).date().isoformat(),
                           'predicted_count': round(point, 2), 'lower_bound': round(min(low, point), 2),
                           'upper_bound': round(max(high, point), 2)})

        # Compare with an equally long, fully observed period ending at the last
        # complete week, since the current week's partial count is excluded.
        baseline_start = origin - timedelta(days=self.horizon_days)
        baseline = sum(float(count) * overlap_fraction(baseline_start, origin,
                       origin - timedelta(weeks=len(data) - i)) for i, count in enumerate(data))
        velocity = ((total - baseline) / baseline * 100) if baseline > 0 else (100.0 if total > 0 else 0.0)
        risk = self.classify_risk(round(total, 2), round(velocity, 1))
        return {'predicted_count': round(total, 2), 'predicted_lower_bound': round(min(lower, total), 2),
                'predicted_upper_bound': round(max(upper, total), 2), 'growth_velocity_percent': round(velocity, 1),
                'risk_level': risk, 'risk_score': round(min(1, max(total / 30, max(0, velocity) / 100)), 4),
                'forecast_points': points}


def build_projection(zones, history_rows, horizon='30d', now=None):
    if horizon not in HORIZONS:
        raise ValueError('horizon must be 7d, 30d, or 90d')
    now = utc_datetime(now)
    history_by_zone = {}
    for row in history_rows:
        history_by_zone.setdefault(row['zone_id'], []).append(row)
    analyses, features = {}, []
    forecaster = ZoneTimeForecaster(HORIZONS[horizon])
    for zone in zones:
        points = regularize_history(history_by_zone.get(zone['id'], []), now)
        forecast = forecaster.forecast_zone([p['actual_count'] for p in points], now)
        analysis = {'region_id': zone['id'], 'zone_id': zone['id'], 'zone_name': zone['zone_name'],
                    'zone_type': zone.get('zone_type', 'h3'), 'report_count': sum(p['actual_count'] for p in points),
                    'historical_points': points, 'region_center_lat': zone['centroid_lat'],
                    'region_center_lng': zone['centroid_lng'], **forecast}
        analyses[zone['id']] = analysis
        features.append({'type': 'Feature', 'id': zone['id'], 'geometry': zone['boundary_geojson'],
                         'properties': {**analysis, 'center_lat': zone['centroid_lat'], 'center_lng': zone['centroid_lng']}})
    return {'time_horizon': horizon, 'horizon': horizon, 'horizon_days': HORIZONS[horizon],
            'model_version': MODEL_VERSION, 'prediction_date': now.isoformat(), 'forecast_date_start': now.isoformat(),
            'forecast_date_end': (now + timedelta(days=HORIZONS[horizon])).isoformat(), 'total_regions': len(zones),
            'hotspot_count': sum(a['risk_level'] == 'high' for a in analyses.values()),
            'total_reports': sum(a['report_count'] for a in analyses.values()), 'region_analyses': analyses,
            'geojson': {'type': 'FeatureCollection', 'features': features}, 'heatmap_geojson': None}
