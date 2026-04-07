import tensorflow as tf
import os


def get_data_generators(raw_dir, img_size=(224, 224), batch_size=32):
    """
    Creates separate train and validation data generators.
    
    KEY FIX: Validation data must NEVER be augmented.
    Augmenting validation data gives falsely optimistic/pessimistic results
    and prevents you from diagnosing overfitting correctly.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # Detect number of class folders
    if os.path.exists(raw_dir):
        subdirs = [
            f for f in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, f))
        ]
        num_classes = len(subdirs)
    else:
        subdirs = []
        num_classes = 2

    print(f"Data Loader detected {num_classes} class folders: {subdirs}")

    # For 2 classes use binary mode (1 sigmoid neuron)
    # For >2 classes use categorical (softmax)
    class_mode = "binary" if num_classes <= 2 else "categorical"

    # --- Training Generator: WITH augmentation ---
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNet-specific normalization
        validation_split=0.2,
        # Realistic augmentations for face mask detection
        rotation_range=20,        # Faces can be slightly tilted
        zoom_range=0.15,          # Slight zoom variation
        width_shift_range=0.1,    # Small horizontal shifts
        height_shift_range=0.1,   # Small vertical shifts
        horizontal_flip=True,     # Mirrors of faces are valid
        brightness_range=[0.8, 1.2],  # Lighting variation
        fill_mode="nearest"
    )

    # --- Validation Generator: NO augmentation (only preprocessing) ---
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="training",
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        raw_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="validation",
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator


if __name__ == "__main__":
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(base_proj_dir, "data", "raw")

    print(f"Testing data loaders on: '{test_dir}'...")
    try:
        train_gen, val_gen = get_data_generators(test_dir, batch_size=32)
        print(f"Train batches : {len(train_gen)}")
        print(f"Val batches   : {len(val_gen)}")
        print(f"Class mapping : {train_gen.class_indices}")
        print("Data generators OK!")
    except Exception as e:
        print(f"Error: {e}")
