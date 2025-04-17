from typing import List, Dict, Tuple
import time
import torch


list_profile_tensor_types = ['float', 'c10::Half', 'c10::BFloat16']
dict_profile2torchdtype = {
    'float': torch.float32,
    'c10::Half': torch.float16,
    'c10::BFloat16': torch.bfloat16,
}

from dataclasses import dataclass
@dataclass
class TensorCfg:
    """
    A class to represent a dummy tensor.
    """
    shape: List[int]
    dtype: str
    strides: List[int]

def build_tensor(tensor_cfg: TensorCfg, device: str = 'cuda') -> torch.Tensor:
    """
    Build a tensor from the dummy tensor.
    
    Args:
        tensor_cfg (TensorCfg): The dummy tensor.
    
    Returns:
        torch.Tensor: The built tensor.
    """
    # random normally distributed tensor
    dtype = dict_profile2torchdtype[tensor_cfg.dtype]
    tensor = torch.randn(tensor_cfg.shape, dtype=dtype, device=device)
    tensor = tensor.as_strided(size=tensor_cfg.shape, stride=tensor_cfg.strides)
    return tensor

def summarize_tensor(tensor: torch.Tensor) -> str:
    """
    Summarize the tensor information.
    
    Args:
        tensor (torch.Tensor): The tensor to summarize.
    
    Returns:
        str: The summary string.
    """
    return f"Tensor(shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, strides={tensor.stride()})"

def benchmark_func(func, device, warmup=50, avg_steps=100):
    """
    Benchmark a function with warmup and average steps.
    Disclaimer: This method would be innacurate for very short ops.
    Args:
        func (callable): The function to benchmark.
        warmup (int): Number of warmup iterations.
        avg_steps (int): Number of iterations to average over.
    Returns:
        float: Average time taken per iteration in microseconds.
    """
    # Warmup phase
    for _ in range(warmup):
        func()

    # Benchmarking phase
    torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(avg_steps):
        func()
    torch.cuda.synchronize(device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_sec = elapsed_time / avg_steps
    avg_time_us = avg_time_sec * 1e6

    return avg_time_us