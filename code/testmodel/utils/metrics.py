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


## Compuational cost ???

def custom_entropy_formula(predictions, num_classes:int=2) -> float:
    # Ensure that values are between 1e-9 and 1 - 1e-9 to avoid log domain errors
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    entropy = -np.sum(predictions * np.log(predictions), axis=1)

    # Maximum possible entropy when each class has equal probability
    # num_classes = predictions.shape[1]
    max_entropy = np.log(num_classes)

    return np.mean(entropy / max_entropy)

def custom_entropy_formula_analysis(predictions, num_classes:int=2) -> float:
    # Ensure predictions are within the valid range for logarithms
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    
    # If predictions are of shape (n, 1), assume binary classification and calculate the complementary probabilities
    if predictions.shape[1] == 1:
        complementary_predictions = 1 - predictions
        predictions = np.hstack((complementary_predictions, predictions))

    # Normalize predictions to ensure they sum to 1 (if necessary)
    predictions /= np.sum(predictions, axis=1, keepdims=True)
    
    # Calculate entropy
    entropy = -np.sum(predictions * np.log(predictions), axis=1)

    # Since this is a binary classification, max entropy is log(2)
    max_entropy = np.log(num_classes)
    normalized_entropy = np.mean(entropy) / max_entropy

    return normalized_entropy

def compute_metrics(outputs, targets, threshold=0.5) -> tuple[float, float, float, float, np.array, np.array, np.array, np.array]:  # type: ignore
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
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


def compare_tensors(tensor_pair1, tensor_pair2) -> bool:
    # retrieve pairs
    outputs_tensor1, targets_tensor1 = tensor_pair1
    outputs_tensor2, targets_tensor2 = tensor_pair2
    
    # compare output & target
    outputs_equal = torch.equal(outputs_tensor1, outputs_tensor2)
    targets_equal = torch.equal(targets_tensor1, targets_tensor2)
    
    return outputs_equal and targets_equal
