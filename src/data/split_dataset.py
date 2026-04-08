import os
import shutil
import random
from pathlib import Path

def split_dataset(raw_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Physically splits the dataset from raw_dir into output_dir/train, output_dir/val, and output_dir/test.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Ratios must sum to 1.0")

    # Ensure output structure
    for split in ['train', 'val', 'test']:
        for category in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, category)):
                os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

    categories = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Splitting categories: {categories}")
    
    for category in categories:
        cat_path = os.path.join(raw_dir, category)
        images = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
        random.shuffle(images)
        
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        
        train_idxs = images[:num_train]
        val_idxs = images[num_train:num_train+num_val]
        test_idxs = images[num_train+num_val:]
        
        splits = {
            'train': train_idxs,
            'val': val_idxs,
            'test': test_idxs
        }
        
        for split_name, split_files in splits.items():
            print(f"  Moving {len(split_files)} images to {split_name}/{category}...")
            for f in split_files:
                src = os.path.join(cat_path, f)
                dst = os.path.join(output_dir, split_name, category, f)
                shutil.copy2(src, dst) # Use copy instead of move to keep raw data intact

    print("\nDataset split complete!")

if __name__ == "__main__":
    from src.utils import load_config
    config = load_config()
    data_conf = config.get("data", {})
    
    raw = data_conf.get("raw_dir", "data/raw")
    processed = data_conf.get("processed_dir", "data/processed")
    
    split_dataset(raw, processed)
