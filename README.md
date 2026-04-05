# Face Mask Detection

A production-ready Deep Learning project to detect whether a person is wearing a face mask or not.
Uses TensorFlow/Keras for model training, FastAPI for the backend API, and OpenCV for real-time webcam inference.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix
   # venv\Scripts\activate   # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional):
   ```bash
   cp .env.sample .env
   ```

## Usage

### 1. Training the Model
To train the CNN model on your dataset (make sure images are in `data/raw/with_mask` and `data/raw/without_mask`):
```bash
python main.py train
```

### 2. Running the API
To start the FastAPI web server:
```bash
python main.py api
```
Access the Swagger UI at `http://localhost:8000/docs`

### 3. Real-Time Webcam Detection
To run OpenCV real-time detection via your webcam:
```bash
python main.py webcam
```
Copy-Item config.dev.yaml config.yaml -Force; python main.py train