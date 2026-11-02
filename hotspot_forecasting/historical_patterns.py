"""Observed density/persistence and separate forward barangay count projections."""
import hashlib
import math
from collections import Counter
from datetime import timedelta
from spatial_analysis import dbscan_labels,compute_convex_hull_polygon,generate_circle_polygon
from config import DBSCAN_EPS_METERS,DBSCAN_MIN_SAMPLES
from zone_forecaster import build_projection,utc_datetime,week_start

VERSION='historical-dbscan-v1'
MAX_REPORTS=5000
WINDOWS={'7d':7,'30d':30,'90d':90}

def temporal_pattern(rows,now):
    periods=[0,0,0]
    for r in rows:
        periods[0 if r['_time']<now-timedelta(days=60) else 1 if r['_time']<now-timedelta(days=30) else 2]+=1
    weeks=len({week_start(r['_time']) for r in rows})
    recent=sum(r['_time']>=now-timedelta(days=7) for r in rows)
    baseline=periods[2]-recent
    if min(periods)>=3 and weeks>=6:
        label='chronic'
    elif periods[1]>=6 and periods[0]<=2 and periods[2]<=2:
        label='transient_spike'
    elif recent>=3 and baseline>=3 and recent/7 >= 2*baseline/23:
        label='recent_spike'
    elif recent>=3 and baseline<3:
        label='recent_concentration'
    elif periods[2]==0:
        label='historical_only'
    else:
        label='intermittent'
    return label,periods,weeks

def clusters(rows):
    labels=dbscan_labels(rows)
    result=[]
    for label in sorted(set(labels)-{-1}):
        members=[row for row,index in zip(rows,labels) if index==label]
        identity='dbscan_'+hashlib.sha256('|'.join(sorted(r['id'] for r in members)).encode()).hexdigest()[:16]
        lat=sum(r['latitude'] for r in members)/len(members)
        lon=sum(r['longitude'] for r in members)/len(members)
        hull=compute_convex_hull_polygon([r['latitude'] for r in members],[r['longitude'] for r in members])
        method='convex_hull' if hull else 'display_circle'
        if hull is None:
            radius=max(50, max(math.hypot((r['latitude']-lat)*111320,(r['longitude']-lon)*111320*math.cos(math.radians(lat))) for r in members)+20)
            hull=generate_circle_polygon(lon,lat,radius)
        result.append({'id':identity,'members':members,'geometry':{'type':'Polygon','coordinates':[hull]},
                       'center_lat':lat,'center_lng':lon,'geometry_method':method})
    return sorted(result,key=lambda c:(-len(c['members']),c['id'])),int(sum(labels==-1))

def build_pattern_analysis(source):
    now=utc_datetime(source['as_of'])
    zones=source['zones']
    codes={z['id'] for z in zones}
    if len(zones)!=30 or len(codes)!=30 or any(not str(c).startswith('1381200') or len(str(c))!=10 or not str(c).isdigit() or not 1<=int(str(c)[-3:])<=30 for c in codes):
        raise ValueError('A complete canonical 30-barangay source is required')
    if not 0<DBSCAN_EPS_METERS<=1000 or not 3<=DBSCAN_MIN_SAMPLES<=100:
        raise ValueError('DBSCAN needs a radius up to 1000 m and at least three samples')
    if len(source['reports'])>MAX_REPORTS:
        raise ValueError('History exceeds the processing limit; no points were sampled')
    names={z['id']:z['zone_name'] for z in zones}
    rows=[];seen=set()
    for raw in source['reports']:
        row={k:raw[k] for k in ('id','created_at','barangay_code','latitude','longitude')}
        row['_time']=utc_datetime(row['created_at'])
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen or row['barangay_code'] not in codes or not all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) for k in ('latitude','longitude')) or not -90<=row['latitude']<=90 or not -180<=row['longitude']<=180 or not now-timedelta(days=90)<=row['_time']<now:
            raise ValueError('Invalid, duplicate, out-of-window or unreconciled report in source')
        seen.add(row['id']);rows.append(row)
    rows.sort(key=lambda r:r['id'])
    quality={k:source['quality'][k] for k in ('eligible_reports_90','included_reports_90','excluded_unreconciled_90','excluded_invalid_location_90')}
    if any(not isinstance(v,int) or v<0 for v in quality.values()) or quality['included_reports_90']!=len(rows) or quality['eligible_reports_90']!=sum(quality[k] for k in ('included_reports_90','excluded_unreconciled_90','excluded_invalid_location_90')):
        raise ValueError('Source coverage counters are inconsistent')
    anchors,noise=clusters(rows)
    member_anchor={r['id']:c['id'] for c in anchors for r in c['members']}
    patterns=[]
    for cluster in anchors:
        members=cluster['members'];label,periods,weeks=temporal_pattern(members,now)
        counts=Counter(r['barangay_code'] for r in members)
        patterns.append({'id':cluster['id'],'pattern':label,'period_counts':periods,'active_weeks':weeks,
                         'report_count':len(members),'observed_counts':{key:sum(r['_time']>=now-timedelta(days=days) for r in members) for key,days in WINDOWS.items()},
                         'barangays':[{'psgc_code':code,'name':names[code],'report_count':count} for code,count in sorted(counts.items())],
                         **{k:cluster[k] for k in ('geometry','center_lat','center_lng','geometry_method')}})
    pattern_by_id={p['id']:p for p in patterns}
    windows={}
    for key,days in WINDOWS.items():
        subset=[r for r in rows if r['_time']>=now-timedelta(days=days)]
        groups,window_noise=(anchors,noise) if days==90 else clusters(subset)
        features=[]
        for cluster in groups:
            anchor_ids=sorted({member_anchor[r['id']] for r in cluster['members'] if r['id'] in member_anchor})
            label=pattern_by_id[anchor_ids[0]]['pattern'] if len(anchor_ids)==1 else 'mixed_membership'
            counts=Counter(r['barangay_code'] for r in cluster['members'])
            features.append({'type':'Feature','geometry':cluster['geometry'],'properties':{
                'cluster_id':cluster['id'],'anchor_ids':anchor_ids,'pattern':label,'report_count':len(cluster['members']),
                'center_lat':cluster['center_lat'],'center_lng':cluster['center_lng'],'geometry_method':cluster['geometry_method'],
                'barangays':[{'psgc_code':code,'name':names[code],'report_count':count} for code,count in sorted(counts.items())]}})
        windows[key]={'lookback_days':days,'start':(now-timedelta(days=days)).isoformat(),'end':now.isoformat(),
                      'report_count':len(subset),'cluster_count':len(features),'noise_count':window_noise,
                      'geojson':{'type':'FeatureCollection','features':features}}
    history=source['weekly_history']
    origin=week_start(now)
    history_keys=set()
    for row in history:
        start=utc_datetime(row['week_start']);key=(row['zone_id'],start)
        if row['zone_id'] not in codes or start!=week_start(start) or not origin-timedelta(weeks=52)<=start<origin or type(row['report_count']) is not int or row['report_count']<0 or key in history_keys:
            raise ValueError('Invalid or duplicate complete-week forecast history')
        history_keys.add(key)
    forecasts={key:build_projection(zones,history,key,now) for key in WINDOWS}
    barangays=[{'psgc_code':z['id'],'name':z['zone_name'],
                'observed_counts':{key:sum(r['barangay_code']==z['id'] and r['_time']>=now-timedelta(days=days) for r in rows) for key,days in WINDOWS.items()},
                'pattern_ids':[p['id'] for p in patterns if any(b['psgc_code']==z['id'] for b in p['barangays'])]} for z in zones]
    return {'schema_version':VERSION,'generated_at':now.isoformat(),'history_start':(now-timedelta(days=90)).isoformat(),
            'algorithm':'DBSCAN (haversine)','params':{'eps_meters':DBSCAN_EPS_METERS,'min_samples':DBSCAN_MIN_SAMPLES},
            'classification_rules':{'version':'recurrence-v1','chronic':'All three non-overlapping 30-day bins have at least 3 reports, across at least 6 UTC weeks.',
             'transient_spike':'Middle 30-day bin has at least 6 reports; both adjacent bins have at most 2.',
             'recent_spike':'Past 7 days have at least 3 reports and twice the daily rate of the preceding 23 days (baseline at least 3).',
             'recent_concentration':'Past 7 days have at least 3 reports, with an insufficient preceding 23-day baseline.'},
            'quality':quality,'windows':windows,'patterns':patterns,'barangays':barangays,'forecasts':forecasts}
