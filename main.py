import argparse
import uvicorn
import os
import cv2
import numpy as np
import tensorflow as tf
from src.pipelines.model_training import train_pipeline

def run_api():
    """Start the FastAPI application."""
    print("\n🚀 Starting server at http://127.0.0.1:8000")
    print("   If you see 404, please ensure you use http:// (not https://)\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

def run_training():
    """Start the model training pipeline."""
    train_pipeline()

def run_webcam():
    """Start real-time webcam detection."""
    try:
        model_path = "models/mask_detector_finetuned.h5"
        if not os.path.exists(model_path):
            model_path = "models/mask_detector.h5"
            
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Tip: Ensure you are running from the project root and TensorFlow is installed correctly.")
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Add 20% padding to the face crop to help ResNet see more context
            padding = int(w * 0.2)
            y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
            x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
            
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            
            # Use ResNet-specific preprocessing
            from tensorflow.keras.applications.resnet50 import preprocess_input
            face_img = preprocess_input(face_img.astype(np.float32))

            preds = model.predict(face_img, verbose=0)[0]
            score = preds[0]
            
            # Final flip: score > 0.5 means NO MASK
            if score > 0.5:
                label = "No Mask"
                conf = score * 100
                color = (0, 0, 255) # Red
            else:
                label = "Mask"
                conf = (1 - score) * 100
                color = (0, 255, 0) # Green

            cv2.putText(frame, f"{label} ({conf:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Face Mask Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection Project")
    parser.add_argument("mode", choices=["api", "train", "webcam"], help="Mode to run the application in")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "train":
        run_training()
    elif args.mode == "webcam":
        run_webcam()
