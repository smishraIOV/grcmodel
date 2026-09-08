"""Device selection: prefer Apple Silicon MPS, fall back to CPU."""

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"selected device: {device}")
