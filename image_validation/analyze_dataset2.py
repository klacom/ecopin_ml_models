import pandas as pd
import numpy as np

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')
df['empty_count'] = df.isnull().sum(axis=1)

print('=== IMAGE ID PATTERNS ===')
df['id_prefix'] = df['image_id'].str.split('_').str[0]
print(df['id_prefix'].value_counts(dropna=False))

for prefix in ['WST', 'POL', 'FLD', 'NENV']:
    subset = df[df['id_prefix'] == prefix]
    if len(subset) > 0:
        print('\n=== ' + prefix + ' ===')
        print('Count:', len(subset))
        print('ID range:', subset['image_id'].iloc[0], 'to', subset['image_id'].iloc[-1])
        print('Primary labels:', subset['primary_label'].value_counts(dropna=False).to_dict())
        print('Empty count: mean=%.1f' % subset['empty_count'].mean(), ', >5 =', (subset['empty_count']>5).sum())
        print('Source URLs (first 5):')
        for i, url in enumerate(subset['source_url'].head(5)):
            print('  ', url)

print('\n=== ID RANGE CHECK ===')
for prefix in ['WST', 'POL', 'FLD', 'NENV']:
    ids = df[df['id_prefix'] == prefix]['image_id'].tolist()
    if ids:
        nums = [int(x.split('_')[1]) for x in ids]
        print('%s: min=%d, max=%d, count=%d' % (prefix, min(nums), max(nums), len(ids)))

print('\n=== ROWS WITH NaN LABEL ===')
nan_label = df[df['primary_label'].isna()]
print('Count:', len(nan_label))
print('ID prefixes:', nan_label['id_prefix'].value_counts(dropna=False).to_dict())
print('Empty count: mean=%.1f' % nan_label['empty_count'].mean())
print('IDs (first 30):', nan_label['image_id'].head(30).tolist())
print('IDs (last 30):', nan_label['image_id'].tail(30).tolist())

print('\n=== SOURCE URL DOMAINS ===')
def get_domain(url):
    if pd.isna(url) or url == '':
        return 'NaN'
    s = str(url)
    if 'flickr.com' in s:
        return 'Flickr'
    if 'wikimedia.org' in s:
        return 'Wikimedia'
    return 'Other'
df['domain'] = df['source_url'].apply(get_domain)
print(df['domain'].value_counts(dropna=False))

print('\n=== DOMAIN PER PREFIX ===')
print(pd.crosstab(df['id_prefix'], df['domain']))
