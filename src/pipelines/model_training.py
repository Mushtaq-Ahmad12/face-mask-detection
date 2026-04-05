import os
from src.model.resnet import build_resnet_model
from src.data.loader import get_data_generators
from src.model.train import train_model
from src.model.evaluation import evaluate_model, plot_training_history
from src.utils import load_config

def train_pipeline():
    print("Starting Model Training Pipeline...")
    
    # Load config
    config = load_config()
    model_conf = config.get("model", {})
    train_conf = config.get("training", {})
    data_conf = config.get("data", {})
    
    # Configure Hardware Device Boundary
    import tensorflow as tf
    device_opt = train_conf.get("device", "cpu").lower()
    if device_opt == "cpu":
        print("Forcing strictly CPU execution...")
        tf.config.set_visible_devices([], 'GPU')
    elif device_opt == "gpu":
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("WARNING: 'gpu' requested in config but no GPU hardware was detected by TensorFlow! Falling back to CPU...")
        else:
            print(f"Device set to GPU. Available GPUs: {len(gpus)}")
            tf.config.set_visible_devices(gpus, 'GPU')

    raw_dir = data_conf.get("raw_dir", "data/raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"Error: No data found in {raw_dir}. Please add data and try again.")
        return
        
    img_size = (model_conf.get("image_height", 224), model_conf.get("image_width", 224))
    batch_size = train_conf.get("batch_size", 32)
    epochs = train_conf.get("epochs", 10)
    save_path = model_conf.get("model_save_path", "models/mask_detector.h5")
    
    train_gen, val_gen = get_data_generators(raw_dir, img_size=img_size, batch_size=batch_size)
    
    model = build_resnet_model(
        img_width=img_size[1], 
        img_height=img_size[0], 
        channels=model_conf.get("channels", 3)
    )
    
    history = train_model(model, train_gen, val_gen, epochs=epochs, save_path=save_path)
    
    # Evaluation and plotting
    plot_training_history(history)
    evaluate_model(model, val_gen, class_names=data_conf.get("categories", ["with_mask", "without_mask"]))
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    train_pipeline()
