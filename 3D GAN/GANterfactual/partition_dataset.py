import os
import shutil
import random

def create_dirs(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, subdir, 'not_normal'), exist_ok=True)

def partition_data(source_dir, dest_dirs, ratios):
    for category in ['normal', 'not_normal']:
        # List all files in the category directory
        category_path = os.path.join(source_dir, category)
        files = os.listdir(category_path)
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split indices
        total_files = len(files)
        train_idx = int(ratios[0] * total_files)
        val_idx = train_idx + int(ratios[1] * total_files)
        
        # Partition the files
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        # Copy files to the destination directories
        for file in train_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(dest_dirs['train'], category, file))
        
        for file in val_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(dest_dirs['val'], category, file))
        
        for file in test_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(dest_dirs['test'], category, file))

# Define source and destination directories
source_dir = 'data'
dest_dirs = {'train': 'data/train', 'val': 'data/val', 'test': 'data/test'}

# Define partition ratios
ratios = [0.7, 0.2, 0.1]

# Create destination directories
create_dirs('data', ['train', 'val', 'test'])

# Partition the data
partition_data(source_dir, dest_dirs, ratios)