import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib 

def compute_loss(output, target):
    loss_fn = nn.BCELoss()

    return loss_fn(output, target.float())


def compute_loss_ee(outputs, target, weights:list = []):
    loss_fn = compute_loss

    if len(weights) == 0:
        weights = [1.0] * len(outputs)
    elif len(weights) < len(outputs):
        massimo = max(weights)
        weights = [massimo if i >= len(weights) else weights[i] for i in range(len(outputs))]
    elif len(weights) > len(outputs):
        weights = weights[:len(outputs)]

    losses = [loss_fn(output, target) for output in outputs]
    
    return sum([losses[i] * weights[i] for i in range(len(outputs))]) / len(outputs)


def custom_entropy_formula(predictions):
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    entropy = -np.sum(predictions * np.log(predictions), axis=1)
    return np.mean(entropy)

def compute_metrics(outputs, targets, threshold=0.5) -> tuple[float, float, float, float, np.array, np.array, np.array, np.array]:  # type: ignore
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
    
    # Convert predictions into binary labels using the provided threshold
    preds = (outputs >= threshold).astype(int)
    
    aucs = []
    f1_scores = []
    accuracies = []
    recalls = []  # Adding recall metric list
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], outputs[:, i])
        f1 = f1_score(targets[:, i], preds[:, i])
        accuracy = accuracy_score(targets[:, i], preds[:, i])
        recall = recall_score(targets[:, i], preds[:, i]) 
        
        aucs.append(auc)
        f1_scores.append(f1)
        accuracies.append(accuracy)
        recalls.append(recall) 

    # Calculate average metrics
    avg_auc = float(np.mean(aucs))
    avg_f1 = float(np.mean(f1_scores))
    avg_acc = float(np.mean(accuracies))
    avg_recall = float(np.mean(recalls))  
    
    return avg_auc, avg_f1, avg_acc, avg_recall, np.array(aucs), np.array(f1_scores), np.array(accuracies), np.array(recalls)

# Example usage would follow after defining the 'outputs' and 'targets' tensors.

