"""Test/demo fixture only. Contains invented locations, no database access."""
import json
from datetime import datetime,timezone
from test_historical_patterns import source,report,NOW
from historical_patterns import build_pattern_analysis

def fixture():
    rows=[report('chronic-'+str(i),d,code='1381200015' if i%2 else '1381200022') for i,d in enumerate([80,72,65,55,47,39,25,17,9])]
    rows += [report('transient-'+str(i),d,lat=14.60,lon=121.09) for i,d in enumerate([85,82,57,56,55,54,53,52,15])]
    rows += [report('recent-'+str(i),d,lat=14.59,lon=121.07) for i,d in enumerate([1,2,3])]
    rows += [report('noise',2,lat=14.63)]
    result=source(rows)
    # Keep the approved source snapshot close to test execution time.
    delta=datetime.now(timezone.utc)-NOW
    from zone_forecaster import utc_datetime,week_start
    for row in result['reports']:
        row['created_at']=(utc_datetime(row['created_at'])+delta).isoformat()
    result['as_of']=(NOW+delta).isoformat()
    # Recompute complete UTC-week history after moving the snapshot.
    counts={}
    for row in result['reports']:
        week=week_start(row['created_at'])
        if week<week_start(result['as_of']):
            key=(row['barangay_code'],week.isoformat());counts[key]=counts.get(key,0)+1
    result['weekly_history']=[{'zone_id':k[0],'week_start':k[1],'report_count':v} for k,v in counts.items()]
    return build_pattern_analysis(result)
if __name__=='__main__':
    import sys
    from pathlib import Path
    Path(sys.argv[1]).write_text(json.dumps(fixture(),allow_nan=False),encoding='utf-8')
