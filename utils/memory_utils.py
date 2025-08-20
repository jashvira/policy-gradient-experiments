"""
Memory management utilities for CUDA operations.

Provides clean abstractions for memory cleanup, monitoring, and optimization
across training and inference operations.
"""

import torch
import logging
from typing import Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def cleanup_cuda_cache(
    device: Optional[Union[str, torch.device]] = None,
    synchronize: bool = True,
    log_memory: bool = False
) -> None:
    """
    Clean up CUDA memory cache on specified device.
    
    Args:
        device: Target CUDA device. If None, uses current device.
        synchronize: Whether to synchronize after cleanup.
        log_memory: Whether to log memory stats before/after.
    """
    if not torch.cuda.is_available():
        return
        
    try:
        # Set target device if specified
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            torch.cuda.set_device(device)
        
        # Log memory before cleanup if requested
        if log_memory:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Before cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        # Clean up unused cached memory
        torch.cuda.empty_cache()
        
        # Synchronize if requested
        if synchronize:
            torch.cuda.synchronize()
            
        # Log memory after cleanup if requested
        if log_memory:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"After cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
    except Exception as e:
        logger.warning(f"CUDA cleanup failed: {e}")


def get_model_device(model: torch.nn.Module) -> Optional[torch.device]:
    """Get the device of the first parameter in a model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


@contextmanager
def cuda_memory_cleanup(
    model: Optional[torch.nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    log_memory: bool = False
):
    """
    Context manager that cleans up CUDA memory after the block executes.
    
    Args:
        model: If provided, will clean up on this model's device.
        device: Explicit device to clean up. Ignored if model is provided.
        log_memory: Whether to log memory usage.
    
    Example:
        with cuda_memory_cleanup(inference_model, log_memory=True):
            # Run inference operations
            outputs = model.generate(...)
        # CUDA cache is automatically cleaned up here
    """
    try:
        yield
    finally:
        if model is not None:
            device = get_model_device(model)
        cleanup_cuda_cache(device=device, log_memory=log_memory)


def log_cuda_memory_stats(device: Optional[Union[str, torch.device]] = None, prefix: str = "") -> None:
    """Log current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return
        
    try:
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            torch.cuda.set_device(device)
            
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_cached = torch.cuda.max_memory_reserved() / 1024**3
        
        device_name = str(device) if device else torch.cuda.current_device()
        prefix_str = f"[{prefix}] " if prefix else ""
        
        logger.info(
            f"{prefix_str}GPU {device_name} Memory - Current: {allocated:.2f}GB allocated, {cached:.2f}GB cached | "
            f"Peak: {max_allocated:.2f}GB allocated, {max_cached:.2f}GB cached"
        )
    except Exception as e:
        logger.warning(f"Failed to log CUDA memory stats: {e}")


def log_all_gpu_memory(prefix: str = "") -> None:
    """Log memory stats for all available GPUs."""
    if not torch.cuda.is_available():
        return
        
    for i in range(torch.cuda.device_count()):
        log_cuda_memory_stats(device=f"cuda:{i}", prefix=prefix)


def reset_peak_memory_stats() -> None:
    """Reset peak memory statistics for all GPUs."""
    if not torch.cuda.is_available():
        return
        
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device=i)


def log_gpu_memory_with_roles(train_device: torch.device, inference_model: torch.nn.Module, metrics: dict) -> None:
    """Log memory stats for all GPUs with role labels and add to metrics."""
    if not torch.cuda.is_available():
        return
        
    train_device_idx = train_device.index if hasattr(train_device, 'index') else 0
    inference_device_idx = next(inference_model.parameters()).device.index if hasattr(next(inference_model.parameters()).device, 'index') else 1
    
    for i in range(torch.cuda.device_count()):
        allocated_gb = torch.cuda.memory_allocated(device=i) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(device=i) / 1024**3
        max_allocated_gb = torch.cuda.max_memory_allocated(device=i) / 1024**3
        metrics[f"memory_allocated_gpu{i}_gb"] = allocated_gb
        metrics[f"memory_reserved_gpu{i}_gb"] = reserved_gb
        metrics[f"memory_peak_gpu{i}_gb"] = max_allocated_gb
        
        # Add role labels
        role = ""
        if i == train_device_idx:
            role = " (Training)"
        elif i == inference_device_idx:
            role = " (Inference)"
            
        print(f"GPU{i}{role} Memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {max_allocated_gb:.2f}GB peak")
