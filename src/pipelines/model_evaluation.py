import os
from src.model.loader import load_pretrained_model
from src.data.loader import get_data_generators
from src.model.evaluation import evaluate_model
from src.utils import load_config

def run_evaluation_pipeline():
    print("Starting Model Evaluation Pipeline...")
    
    config = load_config()
    model_conf = config.get("model", {})
    data_conf = config.get("data", {})
    
    raw_dir = data_conf.get("raw_dir", "data/raw")
    model_path = model_conf.get("model_save_path", "models/mask_detector.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Train the model first.")
        return
        
    img_size = (model_conf.get("image_height", 224), model_conf.get("image_width", 224))
    batch_size = model_conf.get("batch_size", 32)
    categories = data_conf.get("categories", ["with_mask", "without_mask"])
    
    print("Loading data...")
    # Get only validation generator
    _, val_gen = get_data_generators(raw_dir, img_size=img_size, batch_size=batch_size)
    
    print("Loading model...")
    model = load_pretrained_model(model_path)
    
    print("Running evaluation...")
    evaluate_model(model, val_gen, class_names=categories)
    print("Evaluation Pipeline completed successfully.")

if __name__ == "__main__":
    run_evaluation_pipeline()
