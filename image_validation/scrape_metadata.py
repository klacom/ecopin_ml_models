import requests
from bs4 import BeautifulSoup
import re
import json
import time
import pandas as pd
from openpyxl import load_workbook

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

LICENSE_MAP = {
    'by-sa/4.0': ('Creative Commons Attribution-Share Alike 4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
    'by-sa/3.0': ('Creative Commons Attribution-Share Alike 3.0', 'https://creativecommons.org/licenses/by-sa/3.0/'),
    'by-sa/2.0': ('Creative Commons Attribution-Share Alike 2.0', 'https://creativecommons.org/licenses/by-sa/2.0/'),
    'by-sa/2.5': ('Creative Commons Attribution-Share Alike 2.5', 'https://creativecommons.org/licenses/by-sa/2.5/'),
    'by/4.0': ('Creative Commons Attribution 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
    'by/3.0': ('Creative Commons Attribution 3.0', 'https://creativecommons.org/licenses/by/3.0/'),
    'by/2.0': ('Creative Commons Attribution 2.0 Generic', 'https://creativecommons.org/licenses/by/2.0/'),
    'by/2.5': ('Creative Commons Attribution 2.5 Generic', 'https://creativecommons.org/licenses/by/2.5/'),
    'by-nc-sa/4.0': ('Creative Commons Attribution-NonCommercial-Share Alike 4.0', 'https://creativecommons.org/licenses/by-nc-sa/4.0/'),
    'by-nc-sa/3.0': ('Creative Commons Attribution-NonCommercial-Share Alike 3.0', 'https://creativecommons.org/licenses/by-nc-sa/3.0/'),
    'by-nc-sa/2.0': ('Creative Commons Attribution-NonCommercial-Share Alike 2.0', 'https://creativecommons.org/licenses/by-nc-sa/2.0/'),
    'by-nc/4.0': ('Creative Commons Attribution-NonCommercial 4.0', 'https://creativecommons.org/licenses/by-nc/4.0/'),
    'by-nc/3.0': ('Creative Commons Attribution-NonCommercial 3.0', 'https://creativecommons.org/licenses/by-nc/3.0/'),
    'by-nc/2.0': ('Creative Commons Attribution-NonCommercial 2.0 Generic', 'https://creativecommons.org/licenses/by-nc/2.0/'),
    'by-nd/4.0': ('Creative Commons Attribution-NoDerivatives 4.0', 'https://creativecommons.org/licenses/by-nd/4.0/'),
    'by-nd/3.0': ('Creative Commons Attribution-NoDerivatives 3.0', 'https://creativecommons.org/licenses/by-nd/3.0/'),
    'by-nd/2.0': ('Creative Commons Attribution-NoDerivatives 2.0 Generic', 'https://creativecommons.org/licenses/by-nd/2.0/'),
    'by-nc-nd/4.0': ('Creative Commons Attribution-NonCommercial-NoDerivatives 4.0', 'https://creativecommons.org/licenses/by-nc-nd/4.0/'),
    'by-nc-nd/2.0': ('Creative Commons Attribution-NonCommercial-NoDerivatives 2.0 Generic', 'https://creativecommons.org/licenses/by-nc-nd/2.0/'),
    'publicdomain/zero/1.0': ('Public Domain CC0 1.0 Universal', 'https://creativecommons.org/publicdomain/zero/1.0/'),
    'publicdomain/mark/1.0': ('Public Domain Mark 1.0', 'https://creativecommons.org/publicdomain/mark/1.0/'),
}

def match_license(url):
    for key, (name, lic_url) in LICENSE_MAP.items():
        if key in url:
            return name, lic_url
    return None, url

def extract_flickr_metadata(url):
    result = {'_source_url': url}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            result['error'] = 'HTTP %d' % resp.status_code
            return result
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')

        og_title = soup.find('meta', property='og:title')
        if og_title:
            result['title'] = og_title.get('content', '').strip()

        og_img = soup.find('meta', property='og:image')
        if og_img:
            result['image_url'] = og_img.get('content', '')

        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            result['description'] = og_desc.get('content', '').strip()[:300]

        lic_links = soup.find_all('a', href=re.compile(r'creativecommons\.org', re.I))
        if lic_links:
            lic_href = lic_links[0].get('href', '')
            lic_name, lic_canonical = match_license(lic_href)
            if lic_name:
                result['license'] = lic_name
            result['license_url'] = lic_canonical if lic_name else lic_href
            if lic_name and ('CC0' in lic_name or 'Public Domain' in lic_name):
                result['attribution_require'] = 'No'
            elif lic_name:
                result['attribution_require'] = 'Yes'
        
        if 'license' not in result and ('All Rights Reserved' in html or 'all rights reserved' in html.lower()):
            result['license'] = 'All Rights Reserved'
            result['attribution_require'] = 'Yes'

        id_match = re.search(r'/(\d{8,})/?$', url)
        if id_match:
            result['photo_id'] = id_match.group(1)

        owner_match = re.search(r'/photos/([^/]+)/', url)
        if owner_match:
            result['owner_username'] = owner_match.group(1)

        # Find author display name - look in structured data or page
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            result['author_display'] = author_meta.get('content', '').strip()
        else:
            # Try to find in ld+json
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string or '')
                    if isinstance(data, dict):
                        if 'author' in data:
                            a = data['author']
                            if isinstance(a, dict) and a.get('name'):
                                result['author_display'] = a['name']
                                break
                except:
                    pass
        
        # Also try to find owner name from page title format "by OWNERNAME | Flickr"
        title_tag = soup.find('title')
        if title_tag and 'author_display' not in result:
            t = title_tag.get_text(strip=True)
            by_match = re.search(r'\|\s*([^|]+)\s*\|\s*Flickr', t)
            if not by_match:
                by_match = re.search(r'\sby\s(.+?)\s*\|', t)
            if by_match:
                result['author_display'] = by_match.group(1).strip()

        # Dimensions
        dims = re.findall(r'"width"\s*:\s*(\d+)\s*,\s*"height"\s*:\s*(\d+)', html)
        if dims:
            largest = max(dims, key=lambda x: int(x[0]) * int(x[1]))
            result['width'] = int(largest[0])
            result['height'] = int(largest[1])
        else:
            # Try og:image:width meta
            w_meta = soup.find('meta', property='og:image:width')
            h_meta = soup.find('meta', property='og:image:height')
            if w_meta and h_meta:
                result['width'] = int(w_meta.get('content', '0') or 0)
                result['height'] = int(h_meta.get('content', '0') or 0)

        # Location: try from meta or description
        lat_meta = soup.find('meta', property='og:latitude')
        lon_meta = soup.find('meta', property='og:longitude')
        if lat_meta and lon_meta:
            result['lat'] = lat_meta.get('content', '')
            result['lon'] = lon_meta.get('content', '')
        
        # Try to extract location name from description / title
        loc_parts = []
        if result.get('description'):
            d = result['description']
            # Common patterns
            for pat in [r'[Ii]n\s+([A-Z][A-Za-z\s,]+)', r'[Aa]t\s+([A-Z][A-Za-z\s,]+)']:
                m = re.search(pat, d)
                if m:
                    loc_parts.append(m.group(1).strip())
                    break
        if loc_parts:
            result['location_hint'] = loc_parts[0]

    except Exception as e:
        result['error'] = '%s: %s' % (type(e).__name__, str(e))
    return result


def extract_wikimedia_metadata(url):
    result = {'_source_url': url}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            result['error'] = 'HTTP %d' % resp.status_code
            return result
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')
        
        lic_link = soup.find('a', href=re.compile(r'creativecommons\.org', re.I))
        if lic_link:
            lic_href = lic_link.get('href', '')
            lic_name, lic_canon = match_license(lic_href)
            if lic_name:
                result['license'] = lic_name
            result['license_url'] = lic_canon if lic_name else lic_href
            if lic_name and ('CC0' in lic_name or 'Public Domain' in lic_name):
                result['attribution_require'] = 'No'
            elif lic_name:
                result['attribution_require'] = 'Yes'
        
        # Dimensions from file info section
        dim_match = re.search(r'(\d[\d,]*)\s*\u00d7\s*(\d[\d,]*)', html)
        if dim_match:
            result['width'] = int(dim_match.group(1).replace(',', ''))
            result['height'] = int(dim_match.group(2).replace(',', ''))
        
        # Author
        creator = soup.find(id='creator')
        if creator:
            result['author_display'] = creator.get_text(strip=True)[:100]
    except Exception as e:
        result['error'] = str(e)
    return result


if __name__ == '__main__':
    # Quick test with more URLs
    tests = [
        'https://www.flickr.com/photos/joegoauk73/23099017192/',
        'https://www.flickr.com/photos/nahemoth/15031664194/',
        'https://www.flickr.com/photos/howardpoon/937027066/',
        'https://www.flickr.com/photos/waferboard/21317435363/',
    ]
    for u in tests:
        m = extract_flickr_metadata(u)
        keys = ['author_display','title','description','license','license_url','photo_id','width','height','location_hint','error']
        print('URL:', u)
        for k in keys:
            if k in m:
                print('  %s: %s' % (k, m[k]))
        print()
        time.sleep(0.8)
