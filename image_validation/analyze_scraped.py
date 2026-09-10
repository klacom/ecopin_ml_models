import json

with open(r'c:\dev\ecopin_image_validation\scraped_metadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Analyze WST patterns
wst_titles = []
fld_titles = []
for mid, meta in data.items():
    prefix = mid.split('_')[0]
    title = meta.get('title', '')
    desc = meta.get('description', '')
    combined = (title + ' ' + desc).lower()
    if prefix == 'WST':
        wst_titles.append((mid, title, desc, combined))
    elif prefix == 'FLD':
        fld_titles.append((mid, title, desc, combined))

print('=== WST SAMPLE TITLES (50 rows) ===')
for mid, t, d, c in wst_titles[:50]:
    lic = data[mid].get('license', '')[:30]
    print(mid, '|', t[:60], '|', lic)

print()
print('=== FLD SAMPLE TITLES (50 rows) ===')
for mid, t, d, c in fld_titles[:50]:
    lic = data[mid].get('license', '')[:30]
    print(mid, '|', t[:60], '|', lic)

# Count license types per prefix
print()
print('=== WST LICENSES ===')
from collections import Counter
wst_lics = Counter()
for mid, meta in data.items():
    if mid.startswith('WST'):
        l = meta.get('license', 'UNKNOWN')
        wst_lics[l] += 1
for l, c in wst_lics.most_common():
    print(c, l)

print()
print('=== FLD LICENSES ===')
fld_lics = Counter()
for mid, meta in data.items():
    if mid.startswith('FLD'):
        l = meta.get('license', 'UNKNOWN')
        fld_lics[l] += 1
for l, c in fld_lics.most_common():
    print(c, l)
