from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data_fetcher import fetch_zone_history
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
    def test_fetches_all_history_pages_from_validated_view(self, get_client):
        history = [{'zone_id': 'stable-zone', 'week_start': '2026-01-05', 'report_count': i} for i in range(2001)]
        tables = {'spatial_zones': [ZONE], 'zone_weekly_report_history': history}
        get_client.return_value.table.side_effect = lambda name: Query(tables[name])
        zones, rows = fetch_zone_history('h3', NOW)
        self.assertEqual(zones, [ZONE])
        self.assertEqual(rows, history)


if __name__ == '__main__':
    unittest.main()
