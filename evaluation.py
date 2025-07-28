import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, criterion, class_names, output_dir, split_name="Test"):
    """Evaluate model with comprehensive metrics"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    loss, correct, total = 0.0, 0, 0

    logger.info(f"Evaluating model on {split_name} set")

    try:
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                images, labels = images.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)

                loss += batch_loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0])

        # Calculate metrics
        accuracy = 100. * correct / total
        avg_loss = loss / len(data_loader)

        logger.info(f"{split_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Generate reports and visualizations
        generate_classification_report(all_labels, all_preds, class_names, output_dir, split_name)
        create_confusion_matrix(all_labels, all_preds, class_names, output_dir, split_name)
        
        if len(class_names) == 2:
            create_roc_curve(all_labels, all_probs, output_dir, split_name)
            create_pr_curve(all_labels, all_probs, output_dir, split_name)

        return accuracy, all_preds, all_labels, all_probs

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise

def generate_classification_report(labels, preds, class_names, output_dir, split_name):
    """Generate and save classification report"""
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    
    logger.info(f"\n{split_name} Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, f"{split_name.lower()}_classification_report.csv"))

def create_confusion_matrix(labels, preds, class_names, output_dir, split_name):
    """Create and save confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split_name.lower()}_confusion_matrix.png"))
    plt.close()

def create_roc_curve(labels, probs, output_dir, split_name):
    """Create and save ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{split_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{split_name.lower()}_roc_curve.png"))
    plt.close()

def create_pr_curve(labels, probs, output_dir, split_name):
    """Create and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{split_name.lower()}_pr_curve.png"))
    plt.close()

def load_and_evaluate_checkpoint(model, checkpoint_path, data_loader, criterion, class_names, output_dir, split_name):
    """Load checkpoint and evaluate model"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
        
        accuracy, preds, labels, probs = evaluate_model(
            model, data_loader, criterion, class_names, output_dir, split_name
        )
        
        return accuracy, preds, labels, probs
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None, None