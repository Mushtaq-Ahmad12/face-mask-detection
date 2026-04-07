import tensorflow as tf
import os


def train_model(
    model,
    train_data,
    val_data,
    epochs=50,
    save_path="models/mask_detector.h5",
    class_weight=None
):
    """
    Phase 1 Training: Train only the classification head (base frozen).
    Uses aggressive early stopping and LR scheduling.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            monitor="val_accuracy",   # Monitor accuracy, not just loss
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,               # Stop if no improvement for 7 epochs
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    print(f"\n{'='*50}")
    print(f"PHASE 1: Training head only for up to {epochs} epochs...")
    print(f"{'='*50}")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight
    )

    return history


def finetune_model(
    model,
    train_data,
    val_data,
    epochs=20,
    save_path="models/mask_detector_finetuned.h5",
    class_weight=None
):
    """
    Phase 2 Training: Fine-tune the unfrozen top layers at a very small LR.
    Call this AFTER train_model() and after calling unfreeze_for_finetuning().
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-8,
            verbose=1
        ),
    ]

    print(f"\n{'='*50}")
    print(f"PHASE 2: Fine-tuning unfrozen layers for up to {epochs} epochs...")
    print(f"{'='*50}")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight
    )

    return history
