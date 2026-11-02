from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data_fetcher import fetch_zone_history, get_supabase_client, fetch_pattern_history
from test_zone_forecaster import NOW, ZONE


class Query:
    def __init__(self, rows):
        self.rows = rows

    def __getattr__(self, name):
        return lambda *args: self

    def range(self, start, end):
        self.page = self.rows[start:end + 1]
        return self

    def execute(self):
        return SimpleNamespace(data=self.page)


class ZoneHistoryTests(unittest.TestCase):
    @patch('data_fetcher.get_supabase_client')
    def test_pattern_history_uses_one_bounded_rpc_snapshot(self, get_client):
        get_client.return_value.rpc.return_value.execute.return_value.data = {'as_of':'snapshot'}
        self.assertEqual(fetch_pattern_history(), {'as_of':'snapshot'})
        get_client.return_value.rpc.assert_called_once_with('read_hotspot_pattern_history', {'p_max_reports':5000})
        get_client.return_value.table.assert_not_called()

    @patch('data_fetcher.SUPABASE_URL', 'https://example.supabase.co')
    @patch('data_fetcher.SUPABASE_SECRET_KEY', 'sb_secret_deployment_test_not_a_real_key')
    def test_client_accepts_modern_secret_key_without_a_network_request(self):
        client = get_supabase_client()
        self.assertIsNotNone(client.table('spatial_zones').select('id'))

    @patch('data_fetcher.get_supabase_client')
    def test_fetches_all_history_pages_from_validated_view(self, get_client):
        history = [{'zone_id': 'stable-zone', 'week_start': '2026-01-05', 'report_count': i} for i in range(2001)]
        tables = {'spatial_zones': [ZONE], 'zone_weekly_report_history': history}
        get_client.return_value.table.side_effect = lambda name: Query(tables[name])
        zones, rows = fetch_zone_history('h3', NOW)
        self.assertEqual(zones, [ZONE])
        self.assertEqual(rows, history)


if __name__ == '__main__':
    unittest.main()
