"""Device and dtype selection. CPU-first, CUDA when it genuinely exists.

The development machine has Intel Iris Xe integrated graphics, no NVIDIA GPU and
no usable ``nvidia-smi``. Everything in Phase 3 is therefore sized for CPU. CUDA
is *detected* rather than assumed, so the same code runs unchanged on a machine
that has one -- but no dependency is added and no code path is written purely to
be able to claim GPU support.

DirectML is deliberately not used. It would be the only route to the Iris Xe,
and it would add a large, Windows-only dependency to accelerate two models that
already run acceptably on CPU at this corpus size.
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_device", "resolve_dtype", "describe_device", "free_model"]


def resolve_device(requested: str = "auto") -> str:
    """Return the torch device string to use.

    ``auto`` picks CUDA when torch reports a usable CUDA device and CPU
    otherwise. An explicit ``cuda`` request on a machine without CUDA is an
    error, not a silent downgrade: someone who asked for the GPU needs to know
    they did not get it, or they will misread an eight-hour run as normal.
    """
    import torch

    normalized = str(requested or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} was requested but torch reports no CUDA "
                "device. Use device: auto to fall back to CPU automatically."
            )
        return normalized
    if normalized == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported device {requested!r}; expected 'auto', 'cpu' or 'cuda[:N]'"
    )


def resolve_dtype(name: str, device: str) -> Any:
    """Map a dtype name to a torch dtype, refusing float16 on CPU.

    CPU float16 matmul is either unimplemented or emulated and much slower than
    float32, so silently honouring it would look like a mysterious slowdown.
    """
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(mapping)}")
    if name == "float16" and device == "cpu":
        raise ValueError(
            "dtype float16 is not usable on CPU (no accelerated kernels; it is "
            "emulated and slower than float32). Use float32 for CPU runs."
        )
    return mapping[name]


def describe_device(device: str) -> dict[str, Any]:
    """Human- and machine-readable device facts, for the statistics report."""
    import torch

    described: dict[str, Any] = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "threads": int(torch.get_num_threads()),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        described["device_name"] = torch.cuda.get_device_name(0)
    return described


def free_model(model: Any) -> None:
    """Drop a model and reclaim its memory before the next stage loads one.

    Explicit rather than left to the garbage collector: the pipeline's peak
    memory is set by whether two models are ever resident at once, and on a
    laptop that is the difference between running and swapping.
    """
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
