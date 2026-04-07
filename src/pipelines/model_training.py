import os
import numpy as np
import tensorflow as tf

from src.model.resnet import build_resnet_model, unfreeze_for_finetuning
from src.data.loader import get_data_generators
from src.model.train import train_model, finetune_model
from src.model.evaluation import evaluate_model, plot_training_history
from src.utils import load_config


def train_pipeline():
    print("\n" + "="*60)
    print("   FACE MASK DETECTION — TRAINING PIPELINE")
    print("="*60 + "\n")

    # ── 1. Load Config ──────────────────────────────────────────
    config = load_config()
    model_conf  = config.get("model", {})
    train_conf  = config.get("training", {})
    data_conf   = config.get("data", {})

    # ── 2. Configure Hardware ────────────────────────────────────
    device_opt = train_conf.get("device", "cpu").lower()
    if device_opt == "cpu":
        print("⚙  Forcing CPU execution...")
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"⚙  GPU detected: {len(gpus)} device(s)")
            tf.config.set_visible_devices(gpus, 'GPU')
        else:
            print("⚠  No GPU found — falling back to CPU.")

    # ── 3. Validate Data ─────────────────────────────────────────
    raw_dir = data_conf.get("raw_dir", "data/raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"✗ No data found in '{raw_dir}'. Add your dataset and try again.")
        return

    # ── 4. Hyperparameters ───────────────────────────────────────
    img_size        = (model_conf.get("image_height", 224), model_conf.get("image_width", 224))
    batch_size      = train_conf.get("batch_size", 32)
    phase1_epochs   = train_conf.get("epochs", 30)           # Head-only training
    phase2_epochs   = train_conf.get("finetune_epochs", 20)  # Fine-tuning
    learning_rate   = train_conf.get("learning_rate", 0.001)
    finetune_lr     = train_conf.get("finetune_lr", 1e-5)
    save_path       = model_conf.get("model_save_path", "models/mask_detector.h5")
    finetune_path   = save_path.replace(".h5", "_finetuned.h5")
    unfreeze_layers = train_conf.get("unfreeze_layers", 20)

    print(f"📁 Data dir    : {raw_dir}")
    print(f"🖼  Image size  : {img_size}")
    print(f"🔢 Batch size  : {batch_size}")
    print(f"🔁 Phase 1 eps : {phase1_epochs}  (LR={learning_rate})")
    print(f"🔁 Phase 2 eps : {phase2_epochs}  (LR={finetune_lr})\n")

    # ── 5. Data Generators ───────────────────────────────────────
    train_gen, val_gen = get_data_generators(raw_dir, img_size=img_size, batch_size=batch_size)

    class_indices       = train_gen.class_indices
    num_detected_classes = len(class_indices)
    print(f"\n✔ Classes detected: {class_indices}")

    # Binary → 1 sigmoid output | Multi-class → N softmax outputs
    model_num_classes = 1 if num_detected_classes == 2 else num_detected_classes

    # ── 6. Class Weights (handles imbalanced datasets) ───────────
    class_weights = None
    if train_conf.get("use_class_weights", True):
        from sklearn.utils import class_weight as cw_utils
        labels = train_gen.classes
        weights = cw_utils.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights = dict(enumerate(weights))
        print(f"📊 Class weights: {class_weights}")

    # ── 7. Build Model ───────────────────────────────────────────
    model, base_model = build_resnet_model(
        model_name    = model_conf.get("name", "resnet50"),
        img_width     = img_size[1],
        img_height    = img_size[0],
        channels      = model_conf.get("channels", 3),
        num_classes   = model_num_classes,
        learning_rate = learning_rate
    )

    # ── 8. Phase 1: Train Head Only ──────────────────────────────
    history1 = train_model(
        model,
        train_gen,
        val_gen,
        epochs       = phase1_epochs,
        save_path    = save_path,
        class_weight = class_weights
    )

    plot_training_history(history1, save_dir="docs/phase1")

    # ── 9. Phase 2: Fine-tune Top Layers ────────────────────────
    print(f"\nUnfreezing top {unfreeze_layers} layers for fine-tuning...")
    model = unfreeze_for_finetuning(
        model,
        base_model,
        num_layers_to_unfreeze = unfreeze_layers,
        learning_rate          = finetune_lr
    )

    history2 = finetune_model(
        model,
        train_gen,
        val_gen,
        epochs       = phase2_epochs,
        save_path    = finetune_path,
        class_weight = class_weights
    )

    plot_training_history(history2, save_dir="docs/phase2")

    # ── 10. Final Evaluation ─────────────────────────────────────
    print("\n" + "="*60)
    print("   FINAL MODEL EVALUATION")
    print("="*60)
    evaluate_model(
        model,
        val_gen,
        class_names = list(class_indices.keys()),
        save_dir    = "docs/evaluation"
    )

    print("\n✅ Pipeline completed successfully!")
    print(f"   Phase 1 model : {save_path}")
    print(f"   Phase 2 model : {finetune_path}")
    print(f"   Reports       : docs/")


if __name__ == "__main__":
    train_pipeline()
