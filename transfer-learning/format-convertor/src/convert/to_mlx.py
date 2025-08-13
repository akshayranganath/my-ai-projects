"""
Converter for PyTorch model to MLX format.
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os
from pathlib import Path
import json

class MLXMNISTModel(nn.Module):
    """MLX version of our MNIST model"""
    def __init__(self):
        super().__init__()
        # MLX Conv2d expects weights in (out_channels, in_channels, kernel_h, kernel_w) format
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
        print("MLX Model layers:")
        print(f"conv1 weight shape: {self.conv1.weight.shape}")
        print(f"conv2 weight shape: {self.conv2.weight.shape}")
        
    def __call__(self, x):
        # Input shape: (batch, height, width, channels)
        # No need to transpose input since we'll handle the weight format instead
        
        x = self.conv1(x)
        x = mx.maximum(x, 0)  # ReLU
        x = self.conv2(x)
        x = mx.maximum(x, 0)  # ReLU
        
        # Max pooling - MLX doesn't have max_pool2d, so we do it manually
        # Current shape: (batch, height, width, channels)
        x = mx.reshape(x, (x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3]))
        x = mx.max(x, axis=(2, 4))
        
        # Flatten: (batch, height, width, channels) -> (batch, features)
        x = mx.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = mx.maximum(x, 0)  # ReLU
        x = self.fc2(x)
        return mx.log_softmax(x, axis=1)

def convert_to_mlx(model_path: str, output_path: str):
    """
    Convert PyTorch model to MLX format.
    
    Args:
        model_path: Path to the PyTorch model state dict
        output_path: Path to save the MLX model
    """
    # Load PyTorch model
    pytorch_state = torch.load(model_path)
    
    # Create MLX model
    mlx_model = MLXMNISTModel()
    
    # Convert parameters
    mlx_params = {}
    
    # Print PyTorch parameters
    print("PyTorch parameters:", pytorch_state.keys())
    
    # Print MLX parameters
    print("MLX parameters:", mlx_model.parameters().keys())
    
    # Create a dictionary to store parameters for each layer
    layer_params = {
        'conv1': {},
        'conv2': {},
        'fc1': {},
        'fc2': {}
    }
    
    # Group parameters by layer
    for pytorch_name, param in pytorch_state.items():
        layer_name, param_type = pytorch_name.split('.')
        param_np = param.detach().cpu().numpy()
        
        # Handle tensor layout differences
        if 'conv' in layer_name and param_type == 'weight':
            # PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
            # MLX: (out_channels, in_channels, kernel_h, kernel_w)
            # Keep the same format but transpose to match MLX's expected layout
            param_np = np.transpose(param_np, (0, 1, 2, 3))
            print(f"Conv weight shape for {layer_name}:")
            print(f"  PyTorch (out,in,h,w): {param.shape}")
            print(f"  MLX (out,in,h,w): {param_np.shape}")
        
        layer_params[layer_name][param_type] = mx.array(param_np)
    
    # Update MLX model parameters
    for layer_name, params in layer_params.items():
        mlx_params[layer_name] = params
    
    # Update model parameters
    mlx_model.update(mlx_params)
    
    # Save model parameters and metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save parameters as a dictionary using numpy's save function
    save_dict = {}
    for layer_name, layer_params in mlx_params.items():
        for param_type, param in layer_params.items():
            # Convert MLX array to numpy array via list
            param_list = param.tolist()
            save_dict[f"{layer_name}/{param_type}"] = np.array(param_list)
    np.savez(output_path, **save_dict)
    
    # Save model metadata
    metadata = {
        "format": "mlx",
        "model_type": "mnist_classifier",
        "architecture": {
            "input_size": [1, 28, 28],
            "output_size": 10,
            "layers": [
                {"name": "conv1", "type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3},
                {"name": "conv2", "type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3},
                {"name": "fc1", "type": "Linear", "in_features": 9216, "out_features": 128},
                {"name": "fc2", "type": "Linear", "in_features": 128, "out_features": 10}
            ]
        }
    }
    
    metadata_path = output_path + '.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def inference_mlx(model_path: str, input_data: torch.Tensor) -> np.ndarray:
    """
    Run inference using MLX ops directly (NHWC + HWIO for conv weights).
    
    Args:
        model_path: Path to the MLX model weights (.npz)
        input_data: PyTorch tensor [N, C, H, W]
    Returns:
        Probabilities as numpy array [N, 10]
    """
    # Load saved parameters
    with np.load(model_path) as data:
        # Convolution weights: stored in PyTorch OIHW, convert to HWIO for mx.conv2d
        conv1_w_oihw = data["conv1/weight"]  # [O, I, H, W]
        conv1_b = data["conv1/bias"]        # [O]
        conv2_w_oihw = data["conv2/weight"]
        conv2_b = data["conv2/bias"]
        
        # mx.conv2d expects weights in OHWI (out, height, width, in)
        conv1_w_ohwi = np.transpose(conv1_w_oihw, (0, 2, 3, 1))  # [O, H, W, I]
        conv2_w_ohwi = np.transpose(conv2_w_oihw, (0, 2, 3, 1))
        
        # Linear weights: stored as [out_features, in_features]
        fc1_w_out_in = data["fc1/weight"]
        fc1_b = data["fc1/bias"]
        fc2_w_out_in = data["fc2/weight"]
        fc2_b = data["fc2/bias"]
        # Transpose to [in_features, out_features] for x @ W
        fc1_w_in_out = np.transpose(fc1_w_out_in, (1, 0))
        fc2_w_in_out = np.transpose(fc2_w_out_in, (1, 0))

    # Convert inputs to NHWC
    x_np = input_data.detach().cpu().numpy()              # [N, C, H, W]
    x_np = np.transpose(x_np, (0, 2, 3, 1))               # [N, H, W, C]
    x = mx.array(x_np)

    # Conv1 (stride 1, no padding)
    y = mx.conv2d(x, mx.array(conv1_w_ohwi), (1, 1), (0, 0), (1, 1), 1)
    y = y + mx.array(conv1_b)
    y = mx.maximum(y, 0)

    # Conv2
    y = mx.conv2d(y, mx.array(conv2_w_ohwi), (1, 1), (0, 0), (1, 1), 1)
    y = y + mx.array(conv2_b)
    y = mx.maximum(y, 0)

    # MaxPool 2x2, stride 2 (manual)
    # y: [N, H, W, C] -> [N, H//2, 2, W//2, 2, C] -> max over axes 2 and 4
    y = mx.reshape(y, (y.shape[0], y.shape[1] // 2, 2, y.shape[2] // 2, 2, y.shape[3]))
    y = mx.max(y, axis=(2, 4))  # [N, H//2, W//2, C]

    # Flatten to [N, features]
    # Transpose NHWC -> NCHW to match PyTorch's flatten order before FC
    y = mx.transpose(y, (0, 3, 1, 2))
    y = mx.reshape(y, (y.shape[0], -1))

    # FC1
    y = mx.matmul(y, mx.array(fc1_w_in_out)) + mx.array(fc1_b)
    y = mx.maximum(y, 0)

    # FC2 -> logits
    y = mx.matmul(y, mx.array(fc2_w_in_out)) + mx.array(fc2_b)

    # Softmax probabilities
    y_np = np.array(y.tolist())
    y_np = np.exp(y_np - y_np.max(axis=1, keepdims=True))
    y_np = y_np / y_np.sum(axis=1, keepdims=True)
    return y_np

if __name__ == '__main__':
    # Example usage
    model_path = 'models/mnist_best.pt'
    mlx_path = 'models/mnist.mlx.npz'
    
    # Convert model
    convert_to_mlx(model_path, mlx_path)
    print(f"Model converted and saved to {mlx_path}")
    
    # Test inference
    test_input = torch.randn(1, 1, 28, 28)
    predictions = inference_mlx(mlx_path, test_input)
    print(f"Test prediction probabilities: {predictions}")
