import tensorflow as tf
import os

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        model_path = os.getenv("MODEL_PATH", "models/mask_detector.h5")
        if os.path.exists(model_path):
            MODEL = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
    return MODEL
