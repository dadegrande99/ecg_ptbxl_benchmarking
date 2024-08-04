import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd


def create_tensors_from_dataframe(csv_path, output_prefix='output_', target_prefix='target_'):
    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Get the output and target columns
    output_columns = [col for col in df.columns if col.startswith(output_prefix)]
    target_columns = [col for col in df.columns if col.startswith(target_prefix)]
    
    # Create the tensors
    outputs_tensor = torch.tensor(df[output_columns].to_numpy())
    targets_tensor = torch.tensor(df[target_columns].to_numpy())
    
    return outputs_tensor, targets_tensor


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
