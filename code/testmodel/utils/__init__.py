import torch

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
        import torch_xla.core.xla_model as xm # type: ignore
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


