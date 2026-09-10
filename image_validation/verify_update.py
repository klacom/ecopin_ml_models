import pandas as pd

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')
df['empty_count'] = df.isnull().sum(axis=1)

print('=== POST-UPDATE SUMMARY ===')
print('Total rows:', len(df))
print('Rows with 0 empty cells:', (df['empty_count'] == 0).sum())
print('Rows with 1 empty cell:', (df['empty_count'] == 1).sum())
print('Rows with 2 empty cells:', (df['empty_count'] == 2).sum())
print('Rows with 3-5 empty cells:', ((df['empty_count'] >= 3) & (df['empty_count'] <= 5)).sum())
print('Rows with >5 empty cells:', (df['empty_count'] > 5).sum())

print('\n=== PRIMARY LABELS (after update) ===')
print(df['primary_label'].value_counts(dropna=False))

print('\n=== SUBCATEGORIES per PREFIX (after update) ===')
df['prefix'] = df['image_id'].str.split('_').str[0]
for prefix in ['WST', 'FLD', 'POL', 'NEG']:
    subset = df[df['prefix'] == prefix]
    print('\n%s (n=%d):' % (prefix, len(subset)))
    counts = subset['subcategory'].value_counts(dropna=False)
    for k, v in counts.items():
        print('  %s: %d' % (k, v))

print('\n=== DIFFICULTY ===')
print(df['difficulty'].value_counts(dropna=False))

print('\n=== QUALITY FLAG ===')
print(df['quality_flag'].value_counts(dropna=False))

print('\n=== LICENSE VERIFIED ===')
print(df['license_verified'].value_counts(dropna=False))

print('\n=== APPROVED FOR TRAINING ===')
print(df['approved_for_training'].value_counts(dropna=False))

print('\n=== SOURCE NAME ===')
print(df['source_name'].value_counts(dropna=False))

print('\n=== FIELDS STILL EMPTY (counts) ===')
for col in df.columns:
    if col != 'empty_count' and col != 'prefix':
        empty = df[col].isna().sum()
        if empty > 0:
            print('  %s: %d empty' % (col, empty))

print('\n=== SAMPLE UPDATED ROWS (WST_101-110) ===')
cols = ['image_id','filename','primary_label','subcategory','difficulty','source_name','original_author','license','license_url','download_date','source_id','resolution','quality_flag','location','event_group','license_verified','approved_for_training']
for i in range(100, 110):
    print('\n--- %s ---' % df.iloc[i]['image_id'])
    row = df.iloc[i]
    for c in cols:
        v = row[c]
        if pd.notna(v) and v != '':
            s = str(v)
            if len(s) > 60:
                s = s[:60] + '...'
            print('  %s: %s' % (c, s))

print('\n=== SAMPLE UPDATED ROWS (POL_026-035) ===')
pol_start = df[df['image_id'] == 'POL_026'].index[0]
for i in range(pol_start, pol_start + 10):
    print('\n--- %s ---' % df.iloc[i]['image_id'])
    row = df.iloc[i]
    for c in cols:
        v = row[c]
        if pd.notna(v) and v != '':
            s = str(v)
            if len(s) > 60:
                s = s[:60] + '...'
            print('  %s: %s' % (c, s))

print('\n=== SAMPLE UPDATED ROWS (NEG_051-060) ===')
neg_start = df[df['image_id'] == 'NEG_051'].index[0]
for i in range(neg_start, neg_start + 10):
    print('\n--- %s ---' % df.iloc[i]['image_id'])
    row = df.iloc[i]
    for c in cols:
        v = row[c]
        if pd.notna(v) and v != '':
            s = str(v)
            if len(s) > 60:
                s = s[:60] + '...'
            print('  %s: %s' % (c, s))
