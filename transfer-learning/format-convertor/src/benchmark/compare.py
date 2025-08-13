"""
Benchmark GGUF vs MLX inference performance on MNIST.

Notes:
- Uses existing conversion + inference helpers.
- For simplicity, each call loads weights (includes I/O overhead). This is a baseline.
- Results are indicative; for precise microbenchmarks, refactor to preload models once.
"""
import os
import sys
import json
import time
from typing import Tuple

import torch
from torchvision import datasets, transforms
import numpy as np

# Ensure parent src/ directory is on sys.path so `convert.*` can be imported
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from convert.to_gguf import convert_to_gguf, inference_gguf
from convert.to_mlx import convert_to_mlx, inference_mlx


def ensure_models(script_dir: str) -> Tuple[str, str, str]:
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    pt_path = os.path.join(models_dir, 'mnist_best.pt')
    gguf_path = os.path.join(models_dir, 'mnist.gguf')
    mlx_path = os.path.join(models_dir, 'mnist.mlx.npz')

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing PyTorch model at {pt_path}. Run: python src/train.py")

    if not os.path.exists(gguf_path):
        convert_to_gguf(pt_path, gguf_path)

    if not os.path.exists(mlx_path):
        convert_to_mlx(pt_path, mlx_path)

    return pt_path, gguf_path, mlx_path


def load_mnist(batch_size: int = 32, limit: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    # Limit number of samples for quick benchmark
    if limit is not None and limit < len(test_dataset):
        indices = list(range(limit))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return loader


def benchmark_format(name: str, model_path: str, loader, infer_fn) -> dict:
    # Warmup: run a few batches
    warm_batches = 3
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            infer_fn(model_path, x)
            if i + 1 >= warm_batches:
                break

    # Timed run
    num_samples = 0
    num_correct = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            preds = infer_fn(model_path, x)  # numpy [B,10]
            pred_labels = np.argmax(preds, axis=1)
            num_correct += int((pred_labels == y.numpy()).sum())
            num_samples += x.shape[0]
    elapsed = time.perf_counter() - start

    accuracy = 100.0 * num_correct / max(1, num_samples)
    latency_per_sample_ms = (elapsed / max(1, num_samples)) * 1000.0
    throughput_sps = num_samples / max(1e-9, elapsed)

    return {
        'format': name,
        'num_samples': num_samples,
        'elapsed_sec': elapsed,
        'latency_ms_per_sample': latency_per_sample_ms,
        'throughput_samples_per_sec': throughput_sps,
        'accuracy_percent': accuracy,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.abspath(os.path.join(script_dir, '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    _, gguf_path, mlx_path = ensure_models(script_dir)
    loader = load_mnist(batch_size=32, limit=256)

    print(f"Benchmarking on {sum(b[0].shape[0] for b in loader)} samples...")

    gguf_results = benchmark_format('gguf(json)', gguf_path, loader, inference_gguf)
    print("GGUF results:", gguf_results)

    mlx_results = benchmark_format('mlx(npz)', mlx_path, loader, inference_mlx)
    print("MLX results:", mlx_results)

    results = {
        'gguf': gguf_results,
        'mlx': mlx_results,
    }

    out_path = os.path.join(plots_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()

