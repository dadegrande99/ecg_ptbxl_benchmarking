import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib 

def compute_loss(output, target):
    loss_fn = nn.BCELoss()

    return loss_fn(output, target.float())

def custom_entropy_formula(predictions: np.array) -> np.array: # type: ignore
    return -np.nansum(
        np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0)), axis=1
    ) / np.log(predictions.shape[2])

def compute_metrics(outputs, targets, threshold = 0.5):
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    
    # Convertire le previsioni in etichette binarie utilizzando la soglia fornita
    preds = (outputs >= threshold).astype(int)
    
    aucs = []
    f1_scores = []
    accuracies = []
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], outputs[:, i])
        f1 = f1_score(targets[:, i], preds[:, i])
        accuracy = accuracy_score(targets[:, i], preds[:, i])
        aucs.append(auc)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    avg_auc = float(np.mean(aucs))
    avg_f1 = float(np.mean(f1_scores))
    avg_acc = float(np.mean(accuracies))
    
    return avg_auc, avg_f1, avg_acc, np.array(aucs), np.array(f1_scores), np.array(accuracies)

