import os
import cv2
from PIL import Image
from src.utils import load_config

def run_preprocessing_pipeline():
    print("Starting Data Preprocessing & Validation Pipeline...")
    
    config = load_config()
    data_conf = config.get("data", {})
    raw_dir = data_conf.get("raw_dir", "data/raw")
    
    if not os.path.exists(raw_dir):
        print(f"Error: Data directory {raw_dir} does not exist.")
        return
        
    categories = data_conf.get("categories", ["with_mask", "without_mask"])
    
    corrupt_count = 0
    valid_count = 0
    
    for category in categories:
        cat_path = os.path.join(raw_dir, category)
        if not os.path.exists(cat_path):
            print(f"Warning: Category folder {cat_path} not found.")
            continue
            
        print(f"Checking images in {category}...")
        for filename in os.listdir(cat_path):
            file_path = os.path.join(cat_path, filename)
            
            # Skip non-files
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Try opening with PIL
                with Image.open(file_path) as img:
                    img.verify() # verify that it is an image
                    
                # Try reading with cv2 to ensure it's not a truncated/broken format
                cv2_img = cv2.imread(file_path)
                if cv2_img is None:
                    raise Exception("cv2 could not read image.")
                    
                valid_count += 1
            except Exception as e:
                print(f"Corrupt or invalid image found and removed: {file_path}")
                os.remove(file_path)
                corrupt_count += 1
                
    print("\n--- Preprocessing Summary ---")
    print(f"Valid Images: {valid_count}")
    print(f"Corrupt Images Removed: {corrupt_count}")
    print("Data Preprocessing Pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
