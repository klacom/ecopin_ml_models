import pandas as pd
import numpy as np

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')
print('Total rows:', len(df))
print('Columns:', list(df.columns))

df['empty_count'] = df.isnull().sum(axis=1)
df['has_source_url'] = df['source_url'].notna() & (df['source_url'] != '')

df_with_url = df[df['has_source_url']].copy()
print(f'\nRows with source_url: {len(df_with_url)}')
print(f'Empty count distribution:')
print(df_with_url['empty_count'].describe())
many_empty = df_with_url[df_with_url['empty_count'] > 5]
medium_empty = df_with_url[(df_with_url['empty_count'] >= 3) & (df_with_url['empty_count'] <= 5)]
few_empty = df_with_url[df_with_url['empty_count'] < 3]
print(f'\nRows with many empty cells (>5): {len(many_empty)}')
print(f'Rows with medium empty cells (3-5): {len(medium_empty)}')
print(f'Rows with few empty cells (<3): {len(few_empty)}')

print('\n=== PRIMARY LABELS ===')
print(df['primary_label'].value_counts(dropna=False))
print('\n=== SUBCATEGORIES ===')
print(df['subcategory'].value_counts(dropna=False).head(20))
print('\n=== DIFFICULTY ===')
print(df['difficulty'].value_counts(dropna=False))
print('\n=== QUALITY FLAG ===')
print(df['quality_flag'].value_counts(dropna=False))
print('\n=== SOURCE NAME ===')
print(df['source_name'].value_counts(dropna=False))
print('\n=== LICENSE VERIFIED ===')
print(df['license_verified'].value_counts(dropna=False))
print('\n=== APPROVED FOR TRAINING ===')
print(df['approved_for_training'].value_counts(dropna=False))
print('\n=== ATTRIBUTION REQUIRE ===')
print(df['attribution_require'].value_counts(dropna=False))

# Show rows with many empty cells
print('\n=== ROWS WITH MANY EMPTY CELLS (first 20) ===')
cols_show = ['image_id','filename','primary_label','subcategory','difficulty','source_name','source_url','empty_count']
print(many_empty[cols_show].head(20).to_string())
