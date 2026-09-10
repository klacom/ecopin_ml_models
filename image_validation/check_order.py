import pandas as pd

df = pd.read_excel(r'c:\dev\ecopin_image_validation\Ecopin Gantt Chart.xlsx', sheet_name='Dataset')
print('First 10 image_ids:', df['image_id'].head(10).tolist())
print('Rows 95-105:', df['image_id'].iloc[95:105].tolist())
print('Rows 245-255:', df['image_id'].iloc[245:255].tolist())
print('Rows 495-505:', df['image_id'].iloc[495:505].tolist())
print('Rows 745-755:', df['image_id'].iloc[745:755].tolist())
