import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def compute_loss(output, target):
    loss_fn = nn.BCELoss()

    return loss_fn(output, target.float())



def compute_metrics(outputs, targets, threshold=0.5):
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

    avg_auc = np.mean(aucs)
    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(accuracies)
    
    return avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies


def select_device():
    # Check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device selected: {torch.cuda.get_device_name(0)}")
        return device

    # Check for MPS (Apple Silicon GPUs)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device selected")
        return device

    # Check for TPU availability (requires additional setup)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        device = xm.xla_device()
        print("TPU device selected")
        return device
    except ImportError:
        print("PyTorch XLA library not installed, TPU not available.")

    # Check for AMD GPU with ROCm
    if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0):
        device = torch.device("cuda")  # ROCm still uses the cuda flag
        print(f"ROCm device selected: {torch.cuda.get_device_name(0)}")
        return device

    # Use CPU as a fallback if no hardware accelerator is available
    print("No hardware accelerator available, using CPU")
    return torch.device("cpu")
