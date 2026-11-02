import json
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from historical_patterns import build_pattern_analysis
from zone_forecaster import week_start
NOW=datetime(2026,10,29,4,tzinfo=timezone.utc)
ROSTER=json.loads((Path(__file__).resolve().parents[2]/'ecopin_backend_node/data/pasig-barangay-roster.json').read_text(encoding='utf-8'))['barangays']
def report(i,days,code='1381200015',lat=14.57,lon=121.08):
 return {'id':str(i),'created_at':(NOW-timedelta(days=days)).isoformat(),'latitude':lat,'longitude':lon,'barangay_code':code}
def source(reports):
 zones=[{'id':r['psgc_code'],'psgc_code':r['psgc_code'],'zone_name':r['name'],'zone_type':'barangay','centroid_lat':14.57,'centroid_lng':121.08,
  'boundary_geojson':{'type':'Polygon','coordinates':[[[121.07,14.56],[121.09,14.56],[121.09,14.58],[121.07,14.56]]]}} for r in ROSTER]
 counts={}
 for r in reports:
  week=week_start(r['created_at'])
  if week<week_start(NOW):
   key=(r['barangay_code'],week.isoformat());counts[key]=counts.get(key,0)+1
 return {'as_of':NOW.isoformat(),'zones':zones,'reports':reports,
  'weekly_history':[{'zone_id':key[0],'week_start':key[1],'report_count':value} for key,value in counts.items()],
  'quality':{'eligible_reports_90':len(reports),'included_reports_90':len(reports),'excluded_unreconciled_90':0,'excluded_invalid_location_90':0}}
class PatternTests(unittest.TestCase):
 def test_chronic_recurrence_uses_nonoverlapping_bins_not_nested_windows(self):
  rows=[report(i,d) for i,d in enumerate([80,72,65,55,47,39,25,17,9])]
  result=build_pattern_analysis(source(rows));p=result['patterns'][0]
  self.assertEqual(p['pattern'],'chronic');self.assertEqual(p['period_counts'],[3,3,3]);self.assertGreaterEqual(p['active_weeks'],6)
  self.assertEqual(result['windows']['7d']['cluster_count'],0);self.assertEqual(result['windows']['30d']['report_count'],3)
 def test_transient_spike_requires_an_observed_peak_that_has_faded(self):
  rows=[report(i,d) for i,d in enumerate([85,82,57,56,55,54,53,52,15])]
  self.assertEqual(build_pattern_analysis(source(rows))['patterns'][0]['pattern'],'transient_spike')
 def test_recent_concentration_is_not_falsely_declared_transient(self):
  self.assertEqual(build_pattern_analysis(source([report(i,2) for i in range(3)]))['patterns'][0]['pattern'],'recent_concentration')
 def test_recent_spike_compares_rates_over_unequal_period_lengths(self):
  rows=[report(i,d) for i,d in enumerate([20,15,10,5,3,1])]
  self.assertEqual(build_pattern_analysis(source(rows))['patterns'][0]['pattern'],'recent_spike')
 def test_links_use_member_assignments_and_keep_multi_barangay_footprints(self):
  rows=[report(1,1),report(2,2,'1381200022',lat=14.5701),report(3,3)]
  result=build_pattern_analysis(source(rows));p=result['patterns'][0]
  self.assertEqual({b['psgc_code']:b['report_count'] for b in p['barangays']},{'1381200015':2,'1381200022':1})
  self.assertEqual(result['windows']['7d']['geojson']['features'][0]['properties']['anchor_ids'],[p['id']])
 def test_ids_and_classification_are_reproducible_when_input_is_reordered(self):
  rows=[report(i,d) for i,d in enumerate([80,72,65,55,47,39,25,17,9])]
  self.assertEqual(build_pattern_analysis(source(rows))['patterns'],build_pattern_analysis(source(list(reversed(rows))))['patterns'])
 def test_all_horizons_are_real_forward_forecasts_with_30_canonical_trends(self):
  result=build_pattern_analysis(source([report(i,d) for i,d in enumerate([80,50,20])]))
  self.assertEqual(len(result['barangays']),30);self.assertIn('San Miguel',[r['name'] for r in result['barangays']])
  for horizon,days in [('7d',7),('30d',30),('90d',90)]:
   f=result['forecasts'][horizon];self.assertEqual(f['horizon_days'],days)
   self.assertEqual(datetime.fromisoformat(f['forecast_date_end'])-NOW,timedelta(days=days));self.assertEqual(len(f['region_analyses']),30)
 def test_noise_counts_are_preserved_without_raw_id_or_point_exports(self):
  result=build_pattern_analysis(source([report(1,1),report(2,2,lat=14.62)]))
  self.assertEqual(result['windows']['90d']['noise_count'],2)
  self.assertNotIn('member_ids',json.dumps(result));self.assertNotIn('member_report_ids',json.dumps(result))
 def test_invalid_duplicate_future_and_unknown_barangay_inputs_fail(self):
  for rows in [[report(1,-1)],[report(1,91)],[report(1,1,lat=float('nan'))],[report(1,1,'unknown')],[report(1,1),report(1,1)]]:
   with self.assertRaises(ValueError):build_pattern_analysis(source(rows))
 def test_rejects_duplicate_partial_or_invalid_weekly_history(self):
  for history in [[{'zone_id':'1381200015','week_start':'2026-10-27','report_count':3}],
                  [{'zone_id':'1381200015','week_start':'2026-10-19','report_count':-1}],
                  [{'zone_id':'1381200015','week_start':'2026-10-19','report_count':3}]*2]:
   payload=source([]);payload['weekly_history']=history
   with self.assertRaises(ValueError):build_pattern_analysis(payload)
 def test_empty_data_is_an_empty_observation_not_model_accuracy(self):
  r=build_pattern_analysis(source([]));self.assertEqual(r['patterns'],[]);self.assertEqual(r['windows']['90d']['report_count'],0)
  self.assertNotIn('p_value',json.dumps(r));self.assertNotIn('accuracy',r)
if __name__=='__main__':unittest.main()
