import json
import pandas as pd
from collections import Counter

with open(r'c:\dev\ecopin_image_validation\scraped_metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')

# Check license types of rows NOT approved
not_approved = df[df['approved_for_training'].isna() | (df['approved_for_training'] != 'Yes')]
print('Rows NOT approved:', len(not_approved))
print('Licenses for not-approved:')
lic_counts = Counter()
for _, row in not_approved.iterrows():
    lic = str(row['license']) if pd.notna(row['license']) else 'NaN'
    if len(lic) > 50:
        lic = lic[:50]
    lic_counts[lic] += 1
for l, c in lic_counts.most_common():
    print('  %d: %s' % (c, l))

# Show sample "other" subcategories
print('\n=== WST "other" sample titles ===')
import re
wst_other = df[(df['image_id'].str.startswith('WST')) & (df['subcategory'] == 'other')]
print('WST other count:', len(wst_other))
for _, row in wst_other.head(30).iterrows():
    m = meta.get(row['image_id'], {})
    t = m.get('title', '')
    d = m.get('description', '')
    lic = str(row['license'])[:30]
    s = ('%s | %s | %s | %s' % (row['image_id'], t[:50], d[:40], lic)).encode('cp1252', errors='replace').decode('cp1252')
    print(' ', s)

print('\n=== FLD "other" sample titles ===')
fld_other = df[(df['image_id'].str.startswith('FLD')) & (df['subcategory'] == 'other')]
print('FLD other count:', len(fld_other))
for _, row in fld_other.head(30).iterrows():
    m = meta.get(row['image_id'], {})
    t = m.get('title', '')
    d = m.get('description', '')
    lic = str(row['license'])[:30]
    s = ('%s | %s | %s | %s' % (row['image_id'], t[:50], d[:40], lic)).encode('cp1252', errors='replace').decode('cp1252')
    print(' ', s)

print('\n=== POL "other" sample titles ===')
pol_other = df[(df['image_id'].str.startswith('POL')) & (df['subcategory'] == 'other')]
print('POL other count:', len(pol_other))
for _, row in pol_other.head(30).iterrows():
    m = meta.get(row['image_id'], {})
    t = m.get('title', '')
    d = m.get('description', '')
    lic = str(row['license'])[:30]
    s = ('%s | %s | %s | %s' % (row['image_id'], t[:50], d[:40], lic)).encode('cp1252', errors='replace').decode('cp1252')
    print(' ', s)

# Check POL had a new subcategory construction_debris:
pol_cd = df[(df['image_id'].str.startswith('POL')) & (df['subcategory'] == 'construction_debris')]
print('\nPOL construction_debris count:', len(pol_cd))
for _, row in pol_cd.head().iterrows():
    m = meta.get(row['image_id'], {})
    print(' ', row['image_id'], m.get('title', '')[:60])
