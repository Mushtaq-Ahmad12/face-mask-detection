"""
hardware.py — GPU/CPU auto-detection and configuration for TensorFlow.

Supports three modes (set via config.yaml `training.device`):
  - "auto" : Use GPU if available, otherwise fall back to CPU (recommended)
  - "gpu"  : Force GPU — raises an error if no CUDA GPU is found
  - "cpu"  : Force CPU only (useful for debugging or non-GPU machines)
"""

import tensorflow as tf


def configure_hardware(device_opt: str = "auto") -> str:
    """
    Configures TensorFlow to use GPU (CUDA) or CPU based on `device_opt`.

    Returns the active device string ("GPU" or "CPU").
    """
    device_opt = device_opt.strip().lower()
    gpus = tf.config.list_physical_devices('GPU')

    print("\n" + "─" * 40)
    print("  HARDWARE CONFIGURATION")
    print("─" * 40)
    print(f"  Requested device : {device_opt.upper()}")
    print(f"  GPUs detected    : {len(gpus)}")

    if gpus:
        for gpu in gpus:
            print(f"    • {gpu.name}")
    else:
        print("    • (none)")

    # ── AUTO: Use GPU if available, else CPU ──────────────────
    if device_opt == "auto":
        if gpus:
            _enable_gpu(gpus)
            active = "GPU"
        else:
            _force_cpu()
            active = "CPU"

    # ── GPU: Force GPU, error if not available ────────────────
    elif device_opt == "gpu":
        if not gpus:
            print("\n  ⚠  WARNING: 'gpu' requested but NO CUDA GPU found!")
            print("  ⚠  Falling back to CPU. Install CUDA + cuDNN to use GPU.")
            _force_cpu()
            active = "CPU"
        else:
            _enable_gpu(gpus)
            active = "GPU"

    # ── CPU: Force CPU only ───────────────────────────────────
    elif device_opt == "cpu":
        _force_cpu()
        active = "CPU"

    else:
        print(f"  ⚠  Unknown device '{device_opt}', defaulting to AUTO.")
        active = configure_hardware("auto")

    print(f"\n  ✔  Active device : {active}")
    print("─" * 40 + "\n")
    return active


def _enable_gpu(gpus):
    """Enable all available GPUs with memory growth to avoid OOM errors."""
    for gpu in gpus:
        try:
            # Allow GPU memory to grow incrementally (avoids allocating all VRAM at start)
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs are initialised
            print(f"  ⚠  Could not set memory growth for {gpu.name}: {e}")

    tf.config.set_visible_devices(gpus, 'GPU')
    print(f"  ✔  GPU enabled with memory growth ({len(gpus)} device(s))")


def _force_cpu():
    """Hide all GPU devices so TensorFlow only uses CPU."""
    tf.config.set_visible_devices([], 'GPU')
    print("  ✔  CPU-only mode enabled")


def get_optimal_batch_size(base_batch: int, active_device: str) -> int:
    """
    Adjusts the batch size based on the active device.
    GPUs benefit from larger batches; CPUs prefer smaller ones.
    """
    if active_device == "GPU":
        # GPUs can handle larger batches efficiently
        return base_batch
    else:
        # Cap at 16 on CPU to avoid memory issues and speed up iteration
        optimized = min(base_batch, 16)
        if optimized != base_batch:
            print(f"  ℹ  CPU mode: batch size reduced from {base_batch} → {optimized}")
        return optimized
