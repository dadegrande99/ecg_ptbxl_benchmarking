import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import os


# find all directories in a directory
def list_directories(path) -> list[str]:
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def find_files_with_extension(directory_path, file_extension) -> list[str]:
    matched_files = []
    # Normalize the extension to include the dot if it's not already present
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    # List files in the specified directory
    for file in os.listdir(directory_path):
        # Check if the file ends with the extension and is indeed a file
        if file.endswith(file_extension) and os.path.isfile(os.path.join(directory_path, file)):
            # Add the file to the list
            matched_files.append(file)

    return matched_files


def create_tensors_from_dataframe(csv_path, output_prefix='output_', target_prefix='target_') -> tuple[torch.Tensor, torch.Tensor]:
    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Get the output and target columns
    output_columns = [col for col in df.columns if col.startswith(output_prefix)]
    target_columns = [col for col in df.columns if col.startswith(target_prefix)]
    
    # Create the tensors
    outputs_tensor = torch.tensor(df[output_columns].to_numpy())
    targets_tensor = torch.tensor(df[target_columns].to_numpy())
    
    return outputs_tensor, targets_tensor


def select_device() -> torch.device:
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
