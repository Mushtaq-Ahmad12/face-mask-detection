import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_resnet_model(model_name="resnet50", img_width=224, img_height=224, channels=3, num_classes=1):
    if "152" in model_name.lower():
        from tensorflow.keras.applications import ResNet152
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(img_height, img_width, channels))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, channels))
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Classification head
    if num_classes == 1:
        # binary classification -> sigmoid
        predictions = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        # multi-class classification -> softmax
        predictions = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model
