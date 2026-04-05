import os

def check_dataset():
    """
    Reads the dataset from the 'data/raw' folder, counts the number of valid images 
    in each category folder, and prints out the results.
    """
    # Define our target data directory path
    base_dir = os.path.join("data", "raw")
    
    # Verify that our data directory actually exists before proceeding
    if not os.path.exists(base_dir):
        print(f"Error: The directory '{base_dir}' does not exist.")
        return

    total_images = 0
    print("--- Dataset Summary ---")
    
    # Loop through each item in the base directory
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        
        # Skip hidden files (like .DS_Store), .keep files, or non-directories
        # We only want to look inside actual category folders (like 'with_mask')
        if category.startswith('.') or not os.path.isdir(category_path):
            continue
            
        category_image_count = 0
        
        # Look at every file inside the category folder
        for image_filename in os.listdir(category_path):
            # Skip hidden files and placeholders inside the category folder
            if image_filename.startswith('.') or image_filename == ".keep":
                continue
                
            # If it's a valid file, increment our count
            if os.path.isfile(os.path.join(category_path, image_filename)):
                category_image_count += 1
                
        # Print the number of images found for this specific category
        print(f"Category '{category}': {category_image_count} images")
        
        # Add to the global dataset total
        total_images += category_image_count
        
    # Print the final computed total size of the entire dataset
    print("-----------------------")
    print(f"Total Dataset Images: {total_images}")

# When the script is executed, trigger the check_dataset function directly
if __name__ == "__main__":
    check_dataset()
