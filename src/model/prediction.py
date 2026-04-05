import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses a raw numpy image array for the CNN model.
    Resizes the image, converts to RGB if needed, scales to [0, 1], and adds a batch dimension.
    """
    if len(image.shape) == 2: # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        # Assuming BGR input from cv2.imread by default
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0) # Add batch dimension
    
    return image

def predict_single_image(model: tf.keras.Model, image: np.ndarray, threshold=0.5) -> dict:
    """
    Given a model and raw image, returns the prediction and confidence score.
    Returns:
        dict: {"label": "with_mask" or "without_mask", "confidence": float}
    """
    processed_image = preprocess_image(image)
    
    # Prediction output (assuming sigmoid final layer)
    pred_prob = model.predict(processed_image, verbose=0)[0][0]
    
    if pred_prob > threshold:
        label = "without_mask"
        confidence = float(pred_prob)
    else:
        label = "with_mask"
        confidence = float(1.0 - pred_prob)
        
    return {
        "label": label,
        "confidence": confidence
    }
