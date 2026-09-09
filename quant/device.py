"""Device and dtype selection.

CPU/float64 is the default, deliberately. The zero-friction control test in
tests/ asserts that GRC budgets collapse onto a closed-form benchmark, and
float32 cannot resolve that to a useful tolerance. MPS cannot allocate float64
at all, and at this problem's tensor sizes (a 3-vector of controls plus a
few thousand paths) CPU measures faster than MPS anyway -- kernel-launch
overhead dominates the arithmetic. prefer_accelerator exists for when batch
sizes grow enough for that to stop being true.
"""

import torch


def get_device(prefer_accelerator: bool = False) -> torch.device:
    if prefer_accelerator and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


if __name__ == "__main__":
    device = get_device()
    print(f"selected device: {device} (dtype {get_dtype(device)})")
