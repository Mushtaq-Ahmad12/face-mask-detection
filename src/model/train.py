import os
import tensorflow as tf

def train_model(model, train_data, val_data, epochs=10, save_path="models/mask_detector.h5", class_weight=None):
    """
    Trains the model with callbacks and balance support.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15, # Prevent early stop on epoch 7
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight # Added for imbalance handling
    )
    
    return history
