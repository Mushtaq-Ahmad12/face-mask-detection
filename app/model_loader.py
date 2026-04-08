import tensorflow as tf
import os

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        # Priority: 1. Environment Var, 2. Finetuned model, 3. Base model
        default_path = "models/mask_detector_finetuned.h5"
        if not os.path.exists(default_path):
            default_path = "models/mask_detector.h5"
            
        model_path = os.getenv("MODEL_PATH", default_path)
        
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}...")
                MODEL = tf.keras.models.load_model(model_path)
                print("✔ Model loaded successfully.")
            except Exception as e:
                print(f"✗ Failed to load model at {model_path}: {e}")
        else:
            print(f"✗ Warning: No model found at {model_path}")
    return MODEL
