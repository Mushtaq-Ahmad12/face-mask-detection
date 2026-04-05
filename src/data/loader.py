import tensorflow as tf
import os

def get_data_generators(raw_dir, img_size=(224, 224), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # 20% validation split
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    # Train Generator
    train_generator = datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )
    
    # Validation Generator - do not shuffle so evaluation metric indexing works
    val_generator = datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
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
