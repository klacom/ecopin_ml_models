import requests
from bs4 import BeautifulSoup
import re
import json
import time
import pandas as pd
import os

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
    if not url:
        return None, None
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

        id_match = re.search(r'/(\d{6,})/?$', url)
        if id_match:
            result['photo_id'] = id_match.group(1)

        owner_match = re.search(r'/photos/([^/]+)/', url)
        if owner_match:
            result['owner_username'] = owner_match.group(1)

        # Author: try ld+json structured data
        author_found = False
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                txt = script.string or ''
                data = json.loads(txt)
                a = None
                if isinstance(data, dict) and 'author' in data:
                    a = data['author']
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'author' in item:
                            a = item['author']
                            break
                if isinstance(a, dict) and a.get('name'):
                    result['author_display'] = a['name'].strip()
                    author_found = True
                    break
            except:
                pass
        
        if not author_found:
            # Fallback: try the owner profile URL in the page to extract name from link
            owner_link = soup.find('a', class_=re.compile(r'owner|photographer|user-name', re.I))
            if owner_link:
                n = owner_link.get_text(strip=True)
                if n and len(n) > 1 and len(n) < 60:
                    result['author_display'] = n
                    author_found = True
            
        if not author_found and 'owner_username' in result:
            # Last fallback: use the owner username if display name not found
            result['author_display'] = result['owner_username']

        # Dimensions
        dims = re.findall(r'"width"\s*:\s*(\d+)\s*,\s*"height"\s*:\s*(\d+)', html)
        if dims:
            largest = max(dims, key=lambda x: int(x[0]) * int(x[1]))
            result['width'] = int(largest[0])
            result['height'] = int(largest[1])
        else:
            w_meta = soup.find('meta', property='og:image:width')
            h_meta = soup.find('meta', property='og:image:height')
            if w_meta and h_meta:
                try:
                    result['width'] = int(w_meta.get('content', '0') or 0)
                    result['height'] = int(h_meta.get('content', '0') or 0)
                except:
                    pass

        # Location from meta tags
        lat_meta = soup.find('meta', property='og:latitude')
        lon_meta = soup.find('meta', property='og:longitude')
        if lat_meta and lon_meta and lat_meta.get('content') and lon_meta.get('content'):
            result['lat'] = lat_meta.get('content', '')
            result['lon'] = lon_meta.get('content', '')

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
        
        # Dimensions
        file_info = soup.find(id='mw-imagepage-content')
        dim_match = re.search(r'(\d[\d,]*)\s*\u00d7\s*(\d[\d,]*)', html)
        if dim_match:
            result['width'] = int(dim_match.group(1).replace(',', ''))
            result['height'] = int(dim_match.group(2).replace(',', ''))
        
        # Author
        creator = soup.find(id='creator')
        if creator:
            txt = creator.get_text(strip=True)
            if txt:
                result['author_display'] = txt[:100]
        else:
            # fallback: find the creator span class
            c2 = soup.find(class_='fn') or soup.find(class_='vcard')
            if c2:
                txt = c2.get_text(strip=True)
                if txt:
                    result['author_display'] = txt[:100]

        # Location hint
        coord = soup.find(class_='geo-dms') or soup.find(id='mw-imagepage-section-location')
        if coord:
            # Get parent text for location name
            p = coord.find_parent()
            if p:
                t = p.get_text(strip=True)
                if t:
                    result['location_hint'] = t[:100]

    except Exception as e:
        result['error'] = str(e)
    return result


def main():
    EXCEL_PATH = r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx'
    OUTPUT_JSON = r'c:\dev\ecopin_image_validation\scraped_metadata.json'
    
    # Load checkpoint if exists
    checkpoint = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print('Loaded checkpoint: %d URLs already scraped' % len(checkpoint))
    
    df = pd.read_excel(EXCEL_PATH, sheet_name='Dataset')
    
    # Identify rows that need scraping (any row with empty cells)
    cols = list(df.columns)
    
    scraped_count = 0
    errors = 0
    for idx, row in df.iterrows():
        url = row['source_url']
        image_id = row['image_id']
        
        # Skip if this image_id is already in checkpoint AND no error
        if image_id in checkpoint:
            prev = checkpoint[image_id]
            if 'error' not in prev:
                continue
        
        if pd.isna(url) or url == '':
            continue
        
        url_str = str(url).strip()
        if not url_str.startswith('http'):
            continue
        
        is_flickr = 'flickr.com' in url_str.lower()
        is_wikimedia = 'wikimedia.org' in url_str.lower() or 'wikipedia.org' in url_str.lower()
        
        if is_flickr:
            meta = extract_flickr_metadata(url_str)
        elif is_wikimedia:
            meta = extract_wikimedia_metadata(url_str)
        else:
            continue
        
        checkpoint[image_id] = meta
        scraped_count += 1
        
        if 'error' in meta:
            errors += 1
            msg = ('ERR [%s]: %s' % (image_id, meta['error'])).encode('cp1252', errors='replace').decode('cp1252')
            print(msg)
        else:
            lic = (meta.get('license', 'NO_LIC') or '')[:20]
            dim = '%sx%s' % (meta.get('width', '?'), meta.get('height', '?'))
            title = (meta.get('title', '') or '')[:50]
            msg = 'OK  [%s] %s | lic=%s dim=%s %s' % (image_id, dim, lic, dim, title)
            print(msg.encode('cp1252', errors='replace').decode('cp1252'))
        
        # Save every 20 rows
        if scraped_count % 20 == 0:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            print('-- Checkpoint saved (%d total, %d errors)' % (len(checkpoint), errors))
        
        time.sleep(0.4)
    
    # Final save
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    print('\nDone. Scraped %d new rows. Total: %d. Errors: %d' % (
        scraped_count, len(checkpoint), errors))


if __name__ == '__main__':
    main()
