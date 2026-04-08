import tensorflow as tf
import os

def get_data_generators(train_dir, val_dir, test_dir=None, img_size=(224, 224), batch_size=32):
    """
    Creates separate data generators for train, validation, and optionally test.
    This version expects the data to already be split into separate directories.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # Detect number of classes from train_dir
    if os.path.exists(train_dir):
        subdirs = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
        num_classes = len(subdirs)
    else:
        num_classes = 2

    class_mode = "binary" if num_classes <= 2 else "categorical"
    print(f"Data Loader detected {num_classes} classes at {train_dir}")

    # --- Training Generator: WITH augmentation ---
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    # --- Validation/Test Generator: NO augmentation ---
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False,
        seed=42
    )

    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=42
        )

    return train_generator, val_generator, test_generator
