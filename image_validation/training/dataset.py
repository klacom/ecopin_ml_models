import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Import config
from data_config.dataset_paths_and_split_config import RAW_DIR, PROCESSED_DIR

class EcopinDataset(Dataset):
    """PyTorch Dataset for the Ecopin split CSV.
    The CSV contains a column `relative_image_path` which is either
    `raw/...` or `processed/...`. This class resolves the full path and
    loads the image with PIL.
    """
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path, dtype=str)
        self.transform = transform
        # Map class names to indices
        self.class_to_idx = {
            'flooding': 0,
            'non_environmental': 1,
            'pollution': 2,
            'waste': 3,
        }
        # Verify that all primary_label values are known
        unknown = set(self.df['primary_label']) - set(self.class_to_idx.keys())
        if unknown:
            raise ValueError(f'Unknown class labels found: {unknown}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['relative_image_path']
        prefix, sub = rel_path.split('/', 1)
        base = RAW_DIR if prefix == 'raw' else PROCESSED_DIR
        img_path = os.path.join(base, sub)
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['primary_label']]
        if self.transform:
            image = self.transform(image)
        return image, label
