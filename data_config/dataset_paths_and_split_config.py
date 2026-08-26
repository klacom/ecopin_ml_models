
import os

# Base dataset directory (do not copy data)
BASE_DATA_DIR = r'c:/dev/datasets'
RAW_DIR = os.path.join(BASE_DATA_DIR, 'ecopin_dataset', 'raw')
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, 'data', 'processed')
SPLIT_DIR = os.path.join(BASE_DATA_DIR, 'data', 'splits')

# Split CSV paths
TRAIN_CSV = os.path.join(SPLIT_DIR, 'train.csv')
VAL_CSV = os.path.join(SPLIT_DIR, 'val.csv')
TEST_CSV = os.path.join(SPLIT_DIR, 'test.csv')

# Image size for EfficientNet-B0 (224x224)
IMAGE_SIZE = 224

# Random seed for reproducibility
SEED = 42
