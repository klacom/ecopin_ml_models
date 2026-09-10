import json

with open(r'c:\dev\ecopin_image_validation\scraped_metadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ids = sorted(data.keys())
prefixes = {}
for i in ids:
    p = i.split('_')[0]
    if p not in prefixes:
        prefixes[p] = []
    prefixes[p].append(i)

for p in sorted(prefixes.keys()):
    nums = sorted([int(x.split('_')[1]) for x in prefixes[p]])
    print('%s: %d entries, range %03d-%03d' % (p, len(nums), min(nums), max(nums)))
    # Find gaps
    expected = set(range(min(nums), max(nums)+1))
    actual = set(nums)
    missing = sorted(expected - actual)
    if missing:
        print('  Gaps:', missing[:50], '(first 50)' if len(missing) > 50 else '')
