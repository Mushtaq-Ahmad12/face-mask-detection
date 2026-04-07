import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet152
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def build_resnet_model(
    model_name="resnet50",
    img_width=224,
    img_height=224,
    channels=3,
    num_classes=1,
    learning_rate=0.0001
):
    """
    Builds a ResNet-based transfer learning model with anti-overfitting measures.

    Strategy (Two-Phase Training):
      Phase 1 (called here): Freeze ALL base layers, train only the new head.
                             This warms up the head without destroying pre-trained features.
      Phase 2 (fine-tuning): Unfreeze top N layers and re-train at a very small LR.
                             Call `unfreeze_for_finetuning()` for this.
    """
    if "152" in model_name.lower():
        base_model = ResNet152(
            weights='imagenet',
            include_top=False,
            input_shape=(img_height, img_width, channels)
        )
    else:
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(img_height, img_width, channels)
        )

    # --- Phase 1: Freeze ALL base layers ---
    # This prevents catastrophic forgetting during head warm-up
    base_model.trainable = False

    # --- Classification Head ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Block 1
    x = Dense(512, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Block 2
    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Output
    if num_classes == 1:
        predictions = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print(f"Built {model_name.upper()} model | Head Phase (base frozen)")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    return model, base_model


def unfreeze_for_finetuning(model, base_model, num_layers_to_unfreeze=20, learning_rate=1e-5):
    """
    Phase 2: Unfreeze the top N layers of the base model for fine-tuning.
    Always use a VERY small learning rate here (1e-5 or lower).
    """
    base_model.trainable = True
    # Freeze everything except the last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Re-compile with a much smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=['accuracy']
    )

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Fine-tuning: unfroze last {num_layers_to_unfreeze} base layers")
    print(f"  Trainable params: {trainable_count:,} | LR: {learning_rate}")
    return model
