import argparse
import uvicorn
import cv2
import numpy as np
import tensorflow as tf
from src.pipelines.model_training import train_pipeline

def run_api():
    """Start the FastAPI application."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

def run_training():
    """Start the model training pipeline."""
    train_pipeline()

def run_webcam():
    """Start real-time webcam detection."""
    try:
        model = tf.keras.models.load_model("models/mask_detector.h5")
    except Exception as e:
        print("Model not found. Please train first.")
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
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0) / 255.0

            preds = model.predict(face_img)[0]
            label = "Mask" if preds[0] > 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame, f"{label} ({preds[0]:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
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
