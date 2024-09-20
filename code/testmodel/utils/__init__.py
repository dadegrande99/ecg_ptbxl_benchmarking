import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import os
import ast


# find all directories in a directory
def list_directories(path:str) -> list[str]:
    """
    List all directories in a given path.

    Parameters:
    - path (str): Path to the directory to list.

    Returns:
    - list[str]: List of all directories in the given path.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'Path is not a directory: {path}')
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def find_files_with_extension(directory_path:str, file_extension:str) -> list[str]:
    """
    Find all files with a given extension in a directory.

    Parameters:
    - directory_path (str): Path to the directory to search.
    - file_extension (str): Extension of the files to search for.

    Returns:
    - list[str]: List of all files with the given extension in the directory.
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f'Path is not a directory: {directory_path}')
    
    matched_files = []
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    for file in os.listdir(directory_path):
        if file.endswith(file_extension) and os.path.isfile(os.path.join(directory_path, file)):
            matched_files.append(file)

    return matched_files


def create_tensors_from_dataframe(csv_path:str, output_prefix:str='output_', target_prefix:str='target_') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates PyTorch tensors from a CSV file containing output and target columns.

    Parameters:
    - csv_path (str): Path to the CSV file containing the output and target columns.
    - output_prefix (str): Prefix for the output columns in the CSV file. Default is 'output_'.
    - target_prefix (str): Prefix for the target columns in the CSV file. Default is 'target_'.

    Returns:
    - tuple[torch.Tensor, torch.Tensor]: A tuple containing the output and target tensors.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'File not found: {csv_path}')
    if not csv_path.endswith('.csv'):
        raise ValueError('File must be a CSV file')
    
    df = pd.read_csv(csv_path)
    
    output_columns = [col for col in df.columns if col.startswith(output_prefix)]
    target_columns = [col for col in df.columns if col.startswith(target_prefix)]
    
    to_manage = False
    for col in output_columns:
        if isinstance(df[col][0], str):
            to_manage = True
    
    if not to_manage:
        return torch.tensor(df[output_columns].to_numpy()), torch.tensor(df[target_columns].to_numpy())
    
    # Parse the string representations into actual lists for output columns
    for col in output_columns:
        if isinstance(df[col][0], str):
            df[col] = df[col].apply(ast.literal_eval)
    outputs_list = df[output_columns].values.tolist()
    return torch.tensor(outputs_list, dtype=torch.float32), torch.tensor(df[target_columns].values, dtype=torch.float32)


def select_device() -> torch.device:
    """
    Selects the best available device for PyTorch.

    Returns:
    - torch.device: The selected device.
    """
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
