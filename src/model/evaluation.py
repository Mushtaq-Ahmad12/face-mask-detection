import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def plot_training_history(history, save_dir="docs"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    save_path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def evaluate_model(model, val_data, class_names=['with_mask', 'without_mask'], save_dir="docs"):
    os.makedirs(save_dir, exist_ok=True)
    print("Evaluating model...")
    
    # Generator needs to be reset before prediction
    val_data.reset()
    
    y_true = val_data.classes
    y_pred_probs = model.predict(val_data)
    
    # Check if binary (1 output neuron) or multi-class (multiple output neurons)
    if y_pred_probs.shape[-1] == 1:
        # Binary Classification
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_pred_for_roc = y_pred_probs
    else:
        # Multi-class Classification
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_pred_for_roc = y_pred_probs # Will use probability of the positive class or handle separately
    
    # Output Classification Report
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # 5. ROC AUC Curve
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        auc_path = os.path.join(save_dir, "roc_auc_curve.png")
        plt.savefig(auc_path)
        plt.close()
        print(f"ROC AUC curve saved to {auc_path}")
    except Exception as e:
        print(f"Failed to generate AUC ROC curve: {e}")
