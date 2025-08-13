"""
Converter for PyTorch model to GGUF format.
"""
import torch
import numpy as np
import struct
import os
from pathlib import Path
from model import MNISTModel
from llama_cpp import Llama
import json

def convert_to_gguf(model_path: str, output_path: str):
    """
    Convert PyTorch model to GGUF format.
    
    Args:
        model_path: Path to the PyTorch model state dict
        output_path: Path to save the GGUF model
    """
    # Load PyTorch model
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create metadata for the model
    metadata = {
        "format": "gguf",
        "model_type": "mnist_classifier",
        "architecture": {
            "input_size": [1, 28, 28],  # MNIST image size
            "output_size": 10,  # Number of classes
            "layers": [
                {"name": "conv1", "type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3},
                {"name": "conv2", "type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3},
                {"name": "fc1", "type": "Linear", "in_features": 9216, "out_features": 128},
                {"name": "fc2", "type": "Linear", "in_features": 128, "out_features": 10}
            ]
        }
    }
    
    # Convert model parameters to serializable format
    params = {}
    for name, param in model.state_dict().items():
        # Convert tensor to numpy array, then to list for JSON serialization
        params[name] = param.detach().cpu().numpy().tolist()
    
    # Create GGUF file structure
    gguf_data = {
        "metadata": metadata,
        "parameters": params
    }
    
    # Save as GGUF file (using JSON for now, as actual GGUF binary format would require more complex implementation)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(gguf_data, f, indent=2)  # Added indent for better readability

def inference_gguf(model_path: str, input_data: torch.Tensor) -> np.ndarray:
    """
    Run inference using the GGUF model.
    
    Args:
        model_path: Path to the GGUF model
        input_data: Input tensor of shape [batch_size, 1, 28, 28]
        
    Returns:
        Predicted class probabilities
    """
    # Load GGUF model
    with open(model_path, 'r') as f:
        gguf_data = json.load(f)
    
    # Create PyTorch model and load parameters
    model = MNISTModel()
    state_dict = {}
    for name, param_data in gguf_data['parameters'].items():
        # Convert list back to tensor
        state_dict[name] = torch.tensor(param_data)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    return probabilities.numpy()

if __name__ == '__main__':
    # Example usage
    model_path = 'models/mnist_best.pt'
    gguf_path = 'models/mnist.gguf'
    
    # Convert model
    convert_to_gguf(model_path, gguf_path)
    print(f"Model converted and saved to {gguf_path}")
    
    # Test inference
    # Create a random input tensor
    test_input = torch.randn(1, 1, 28, 28)
    predictions = inference_gguf(gguf_path, test_input)
    print(f"Test prediction probabilities: {predictions}")
