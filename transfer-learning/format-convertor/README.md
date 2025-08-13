# Format Convertor

![./benchmarking.png](./benchmarking.png)


With this project, I want to understand the concept of `gguf` and `mlx` files. Using a simple training algorithm like MNIST, I want to build a trained model. With this model, I'll do the following:

1. Save model to some binary format (like `safetensors`)
2. Convert the model to `gguf`, deploy and test.
3. Convert the model to `mlx`, deploy and test.
4. Lastly run an inference performance comparison between the 2 formats.

I will be running the code on my local Mac machine. So MLX as a format makes sense.

## Project Structure

```
format-convertor/
├── requirements.txt
├── src/
│   ├── model.py           # MNIST model definition
│   ├── train.py           # Training script
│   ├── convert/
│   │   ├── to_gguf.py     # GGUF conversion
│   │   └── to_mlx.py      # MLX conversion
│   └── benchmark/
│       └── compare.py     # Performance comparison
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1) Train the MNIST model

This downloads MNIST, trains the CNN, and saves the best weights to `src/models/` as both `.pt` and `.safetensors`. It also saves a training curve plot under `src/plots/`.

```bash
python src/train.py
```

Artifacts produced:
- `src/models/mnist_best.pt`
- `src/models/mnist_best.safetensors`
- `src/plots/training_curves.png`

### 2) Create and test the GGUF model (demo JSON-based)

We provide a simple JSON-based stand‑in for true GGUF for this MNIST classifier (sufficient for experimentation):

```bash
python src/test_gguf.py
```

This converts the PyTorch model to `src/models/mnist.gguf` and runs sample predictions, saving a preview to:
- `src/plots/gguf_test_predictions.png`

### 3) Create and test the MLX model

Convert PyTorch weights to MLX (`.npz`) and run sample predictions:

```bash
python src/test_mlx.py
```

This creates:
- `src/models/mnist.mlx.npz`
- `src/plots/mlx_test_predictions.png`

### 4) Run the performance comparison

Benchmarks both formats over a subset of the MNIST test set and saves a JSON with metrics.

```bash
python src/benchmark/compare.py
```

This produces:
- `src/plots/benchmark_results.json`

### 5) Read and interpret the results

Open the generated files under `src/plots/`:

- Training dynamics: `src/plots/training_curves.png` shows loss/accuracy across epochs for the PyTorch model.
- Sanity checks: `src/plots/gguf_test_predictions.png` and `src/plots/mlx_test_predictions.png` show predictions on a handful of test images.
- Benchmark summary: `src/plots/benchmark_results.json` contains per‑format metrics, for example:

```json
{
  "gguf": {
    "format": "gguf(json)",
    "num_samples": 256,
    "elapsed_sec": 2.76,
    "latency_ms_per_sample": 10.79,
    "throughput_samples_per_sec": 92.67,
    "accuracy_percent": 98.8
  },
  "mlx": {
    "format": "mlx(npz)",
    "num_samples": 256,
    "elapsed_sec": 0.05,
    "latency_ms_per_sample": 0.20,
    "throughput_samples_per_sec": 4914.8,
    "accuracy_percent": 98.8
  }
}
```

How to interpret:
- **accuracy_percent**: Should be close to the PyTorch model’s test accuracy (~99%). If it’s much lower, verify conversions and input/weight layouts.
- **latency_ms_per_sample** and **throughput_samples_per_sec**: End‑to‑end numbers for this script (include model load/I/O). For pure compute microbenchmarks, preload models once and time only the forward pass.

## References:

- [GGUF](https://huggingface.co/docs/hub/gguf) on HuggingFace
- [MLX](https://opensource.apple.com/projects/mlx/) on Apple
