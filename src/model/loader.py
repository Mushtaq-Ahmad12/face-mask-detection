import os
import tensorflow as tf

def load_pretrained_model(path: str) -> tf.keras.Model:
    """
    Loads a pretrained Keras model from a given path.
    
    Args:
        path (str): The file path to the saved model (e.g., .h5 or SavedModel format).
        
    Returns:
        tf.keras.Model: The loaded Keras model.
        
    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at path: {path}")
        
    print(f"Loading model from {path}...")
    model = tf.keras.models.load_model(path)
    print("Model loaded successfully.")
    return model
