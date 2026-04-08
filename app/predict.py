from fastapi import APIRouter, File, UploadFile, HTTPException
from app.model_loader import get_model
import numpy as np
import cv2

router = APIRouter()

@router.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Use ResNet-specific preprocessing
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = preprocess_input(img.astype(np.float32))
    
    preds = model.predict(img, verbose=0)[0]
    score = float(preds[0])
    
    # Final flip: score > 0.5 is NO_MASK
    label = "no_mask" if score > 0.5 else "mask"
    confidence = score if label == "no_mask" else 1.0 - score
    
    return {
        "prediction": label,
        "confidence": confidence
    }
