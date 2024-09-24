import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def compute_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    This function computes the binary cross-entropy loss between the model outputs and target labels.

    Parameters:
    - output (torch.Tensor): Model outputs.
    - target (torch.Tensor): Target labels.

    Returns:
    - torch.Tensor: Binary cross-entropy loss.
    """
    loss_fn = nn.BCELoss()

    return loss_fn(output, target.float())


def compute_loss_ee(outputs: torch.Tensor, target: torch.Tensor, weights: list = []) -> torch.Tensor:
    """
    This function computes the ensemble entropy loss between the model outputs and target labels.

    Parameters:
    - outputs (list[torch.Tensor]): List of model outputs.
    - target (torch.Tensor): Target labels.
    - weights (list): List of weights for the ensemble entropy loss. Default is an empty list.

    Returns:
    - torch.Tensor: weighted average entropy loss.
    """
    loss_fn = compute_loss

    if len(weights) == 0:
        weights = [1.0] * len(outputs)
    elif len(weights) < len(outputs):
        massimo = max(weights)
        weights = [massimo if i >= len(weights) else weights[i]
                   for i in range(len(outputs))]
    elif len(weights) > len(outputs):
        weights = weights[:len(outputs)]

    losses = [loss_fn(output, target) for output in outputs]

    return sum([losses[i] * weights[i] for i in range(len(outputs))]) / len(outputs)


def custom_entropy_formula(predictions: np.ndarray, num_classes: int = 2) -> float:
    """
    This function calculates the normalized entropy for a given set of predictions.

    Parameters:
    - predictions (np.ndarray): Array of model predictions.
    - num_classes (int): Number of classes. Default is 2.

    Returns:
    - float: Normalized entropy value.
    """
    # Ensure that values are between 1e-9 and 1 - 1e-9 to avoid log domain errors
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    entropy = -np.sum(predictions * np.log(predictions), axis=1)

    # Maximum possible entropy when each class has equal probability
    # num_classes = predictions.shape[1]
    max_entropy = np.log(num_classes)

    return np.mean(entropy / max_entropy)


def custom_entropy_formula_analysis(predictions: np.ndarray, num_classes: int = 2) -> float:
    """
    This function calculates the normalized entropy for a given set of predictions.

    Parameters:
    - predictions (np.ndarray): Array of model predictions.
    - num_classes (int): Number of classes. Default is 2.

    Returns:
    - float: Normalized entropy value.
    """
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


# type: ignore
def compute_metrics(outputs, targets, threshold=0.5) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function computes the average AUC, F1 score, accuracy, and recall for the model outputs and target labels.

    Parameters:
    - outputs (torch.Tensor or np.ndarray): Model outputs.
    - targets (torch.Tensor or np.ndarray): Target labels.
    - threshold (float): Threshold for converting predictions into binary labels. Default is 0.5.

    Returns:
    - tuple[float, float, float, float, np.array, np.array, np.array, np.array]:
        A tuple containing the average AUC, F1 score, accuracy, and recall, 
        as well as the individual AUC, F1 score, accuracy, and recall values for each class.
    """
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


def compare_tensors(tensor_pair1: tuple[torch.Tensor, torch.Tensor],
                    tensor_pair2: tuple[torch.Tensor, torch.Tensor]) -> bool:
    """
    This function compares two pairs of tensors to check if they are equal.

    Parameters:
    - tensor_pair1 (tuple[torch.Tensor, torch.Tensor]): Tuple containing the first pair of tensors.
    - tensor_pair2 (tuple[torch.Tensor, torch.Tensor]): Tuple containing the second pair of tensors.

    Returns:
    - bool: True if the tensors are equal, False otherwise.
    """
    # retrieve pairs
    outputs_tensor1, targets_tensor1 = tensor_pair1
    outputs_tensor2, targets_tensor2 = tensor_pair2

    # compare output & target
    outputs_equal = torch.equal(outputs_tensor1, outputs_tensor2)
    targets_equal = torch.equal(targets_tensor1, targets_tensor2)

    return outputs_equal and targets_equal


def custom_entropy_per_sample(predictions: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    This function calculates the normalized entropy per sample for a given set of predictions.

    Parameters:
    - predictions (np.ndarray): Array of model predictions.
    - num_classes (int): Number of classes. Default is 2.

    Returns:
    - np.ndarray: Array of normalized entropy values per sample.
    """
    # Ensure that values are between 1e-9 and 1 - 1e-9 to avoid log domain errors
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    entropy = -np.sum(predictions * np.log(predictions), axis=1)

    # Maximum possible entropy when each class has equal probability
    max_entropy = np.log(num_classes)

    # Normalized entropy per sample
    normalized_entropy = entropy / max_entropy

    return normalized_entropy  # Returns an array of normalized entropy values per sample


def custom_mcd_entropy_per_sample(predictions: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    This function calculates the normalized entropy per sample for a given set of predictions.

    Parameters:
    - predictions (np.ndarray): Array of model predictions.
    - num_classes (int): Number of classes. Default is 2.

    Returns:
    - np.ndarray: Array of normalized entropy values per sample
    """
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    predictions = np.mean(predictions, axis=2).squeeze()

    # Calculate entropy per sample
    entropy = - predictions * np.log(predictions)

    # Maximum possible entropy when each class has equal probability
    max_entropy = np.log(num_classes)

    # Normalized entropy per sample
    normalized_entropy = entropy / max_entropy

    return normalized_entropy  # Returns an array of normalized entropy values per sample


def variational_ratios(outputs):
    """
    This function calculates the variational ratios for a given set of model outputs.

    Parameters:
    - outputs (torch.Tensor or np.ndarray): Model outputs.

    Returns:
    - float: Variational ratio value.
    """
    # flatten the outputs (is a tensor or numpy array)
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    outputs = outputs.flatten()

    return 1 - np.mean(outputs)


def compute_diff_and_confidence(outputs_targets_tuple: tuple[torch.Tensor, torch.Tensor], mcd: bool = False) -> list[tuple[float, float]]:
    """
    This function computes the difference between the model outputs and target labels, and the confidence of the model predictions.

    Parameters:
    - outputs_targets_tuple (tuple[torch.Tensor, torch.Tensor]): Tuple containing the model outputs and target labels.
    - mcd (bool): Flag to indicate whether the model uses Monte Carlo Dropout, it change the entropy formula used. Default is False.

    Returns:
    - list[tuple[float, float]]: List of tuples containing the difference between the model outputs and target labels, and the confidence of the model predictions.
    """
    if mcd:
        entropy_formula = custom_mcd_entropy_per_sample
    else:
        entropy_formula = custom_entropy_per_sample

    outputs, targets = outputs_targets_tuple
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Calculate confidence as 1 - normalized entropy per sample
    confidence = 1 - entropy_formula(outputs_np)

    if mcd:
        outputs_np = outputs_np.mean(axis=2)
    diff = np.abs(outputs_np - targets_np)

    # Return list of tuples (difference, confidence)
    return list(zip(diff, confidence))


def metrics_per_confidence(outputs_targets_tuple: tuple[torch.Tensor, torch.Tensor], metric: str = "accuracy", mcd: bool = False, step: float = 0.05, threshold: float = 0.5) -> list[tuple[float, float]]:
    """
    This function computes the specified metric for different confidence levels.

    Parameters:
    - outputs_targets_tuple (tuple[torch.Tensor, torch.Tensor]): Tuple containing the model outputs and target labels.
    - metric (str): Metric to compute. Default is "accuracy".
    - mcd (bool): Flag to indicate whether the model uses Monte Carlo Dropout. Default is False.
    - step (float): Step size for confidence levels. Default is 0.05.
    - threshold (float): Threshold for converting predictions into binary labels. Default is 0.5.

    Returns:
    - list[tuple[float, float]]: List of tuples containing the confidence levels and the corresponding metric values.
    """
    # Check if the metric is supported and select the appropriate function
    metric = metric.lower()
    if metric not in ['accuracy', 'auc', 'f1', 'recall']:
        raise ValueError(f"metric {metric} not supported")
    if metric == 'accuracy':
        metric_func = accuracy_score
    elif metric == 'auc':
        metric_func = roc_auc_score
    elif metric == 'f1':
        metric_func = f1_score
    else:
        metric_func = recall_score

    # Select the appropriate entropy formula based on the model type
    if mcd:
        entropy_formula = custom_mcd_entropy_per_sample
    else:
        entropy_formula = custom_entropy_per_sample

    # Generate confidence levels based on the specified step size
    step_check = np.arange(0, 1, step)
    outputs, targets = outputs_targets_tuple
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    confidence = 1 - entropy_formula(outputs_np)
    min_confidence = confidence.min()
    max_confidence = confidence.max()

    if mcd:
        outputs_np = outputs_np.mean(axis=2)

    if outputs_np.ndim == 1:
        outputs_np = outputs_np.reshape(-1, 1)
        targets_np = targets_np.reshape(-1, 1)

    outputs_np = (outputs_np >= threshold).astype(int)
    targets_np = targets_np.astype(int)

    metric_values = []
    for step in step_check:
        covered = confidence <= step
        if step < min_confidence:
            metric_values.append((float(step), 0.0))
        elif step <= max_confidence:
            metric_values.append((float(step), metric_func(
                targets_np[covered], outputs_np[covered])))

    return metric_values
