import tensorflow as tf
import os

from tensorflow.keras.applications.resnet50 import preprocess_input

def get_data_generators(raw_dir, img_size=(224, 224), batch_size=32):
    # Detect number of classes on disk
    if os.path.exists(raw_dir):
        subdirs = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
        num_classes = len(subdirs)
    else:
        num_classes = 2
        
    print(f"Data Loader detected {num_classes} folders: {subdirs if os.path.exists(raw_dir) else 'N/A'}")
    
    # Use categorical for any setup with > 2 folders or if multi-class output is intended
    class_mode = "categorical" if num_classes > 2 else "binary"
    
    # Use standard ResNet preprocessing instead of manual 1/255 scaling
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2, # 20% validation split
        rotation_range=30, # Increased augmentation for better generalization
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    # Train Generator
    train_generator = datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="training",
        shuffle=True
    )
    
    # Validation Generator
    val_generator = datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="validation",
        shuffle=False
    )
    
    return train_generator, val_generator

if __name__ == "__main__":
    # Test block to verify loaders work standalone
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(base_proj_dir, "data", "raw")
    
    print(f"Testing dataloaders targeting '{test_dir}'...")
    try:
        train_gen, val_gen = get_data_generators(test_dir, batch_size=32)
        print("-------------------------------")
        print(f"Total training batches generated: {len(train_gen)}")
        print(f"Total validation batches generated: {len(val_gen)}")
        if train_gen.class_indices:
            print(f"Class Mapping: {train_gen.class_indices}")
        print("-------------------------------")
        print("Data generators successfully initialized!")
    except Exception as e:
        print(f"Encountered an exception loading data: {e}")
