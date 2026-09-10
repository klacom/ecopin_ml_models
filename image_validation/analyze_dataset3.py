import pandas as pd

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')
df['empty_count'] = df.isnull().sum(axis=1)

# Look at completed examples from each source
completed = df[df['empty_count'] <= 2]
print('=== COMPLETED EXAMPLES (few empty cells) ===')
print('Count:', len(completed))

cols = ['image_id','filename','primary_label','subcategory','difficulty','source_name','source_url','original_author','license','license_url','attribution_require','download_date','source_id','resolution','quality_flag','location','event_group','notes','license_verified','approved_for_training']
# Show 5 complete Wikimedia examples
wm = completed[completed['source_name'] == 'Wikimedia Commons']
print('\n=== WIKIMEDIA COMPLETED EXAMPLES (5 rows) ===')
for _, row in wm.head(5).iterrows():
    for c in cols:
        val = row[c]
        if not pd.isna(val) and val != '':
            print('%s: %s' % (c, val))
    print('---')

# Show 2 Flickr examples if any
fl = completed[completed['source_name'] == 'Flickr']
print('\n=== FLICKR COMPLETED EXAMPLES ===')
print('Count:', len(fl))
for _, row in fl.head(5).iterrows():
    for c in cols:
        val = row[c]
        if not pd.isna(val) and val != '':
            print('%s: %s' % (c, val))
    print('---')

# Show incomplete POL (pollution) rows - these have labels but empty metadata
pol_inc = df[(df['image_id'].str.startswith('POL')) & (df['empty_count'] > 5)]
print('\n=== POL INCOMPLETE (first 10) ===')
print(pol_inc[['image_id','source_url','empty_count']].head(10).to_string())

# Show incomplete NEG (non_environmental) rows
neg_inc = df[(df['image_id'].str.startswith('NEG')) & (df['empty_count'] > 5)]
print('\n=== NEG INCOMPLETE (first 10) ===')
print(neg_inc[['image_id','source_url','empty_count']].head(10).to_string())
