import json

with open(r'c:\dev\ecopin_image_validation\scraped_metadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== WST FLICKR TITLES (WST_101-150) ===')
for i in range(101, 151):
    mid = 'WST_%03d' % i
    if mid in data:
        meta = data[mid]
        t = meta.get('title', '')
        d = meta.get('description', '')
        lic = meta.get('license', '')[:35]
        w = meta.get('width', '?')
        h = meta.get('height', '?')
        author = meta.get('author_display', '')[:30]
        print('%s | %-55s | %s | %sx%s | %s' % (mid, t[:55], lic, w, h, author))
    else:
        print('%s | NOT SCRAPED' % mid)

print()
print('=== FLD FLICKR TITLES (FLD_101-150) ===')
for i in range(101, 151):
    mid = 'FLD_%03d' % i
    if mid in data:
        meta = data[mid]
        t = meta.get('title', '')
        d = meta.get('description', '')
        lic = meta.get('license', '')[:35]
        w = meta.get('width', '?')
        h = meta.get('height', '?')
        author = meta.get('author_display', '')[:30]
        print('%s | %-55s | %s | %sx%s | %s' % (mid, t[:55], lic, w, h, author))
    else:
        print('%s | NOT SCRAPED' % mid)
