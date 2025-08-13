"""
Test script for MLX model conversion and inference.
"""
import torch
from torchvision import datasets, transforms
from convert.to_mlx import convert_to_mlx, inference_mlx
import matplotlib.pyplot as plt
import numpy as np
import os

def test_mlx_model():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Convert model to MLX
    model_path = os.path.join(models_dir, 'mnist_best.pt')
    mlx_path = os.path.join(models_dir, 'mnist.mlx.npz')
    
    # Print paths for debugging
    print(f"Loading PyTorch model from: {model_path}")
    print(f"Saving MLX model to: {mlx_path}")
    
    convert_to_mlx(model_path, mlx_path)
    print(f"Model converted and saved to {mlx_path}")
    
    # Verify file exists
    if not os.path.exists(mlx_path):
        raise FileNotFoundError(f"MLX model file not found at {mlx_path}")
    
    # Test on a few examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, (data, target) in enumerate(test_loader):
        if i >= 10:  # Test first 10 images
            break
            
        # Get prediction
        pred = inference_mlx(mlx_path, data)
        predicted_class = np.argmax(pred)
        
        # Plot image and prediction
        axes[i].imshow(data[0, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'True: {target.item()}\nPred: {predicted_class}')
    
    plt.tight_layout()
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'mlx_test_predictions.png'))
    plt.close()
    print("Test predictions saved to plots/mlx_test_predictions.png")

if __name__ == '__main__':
    test_mlx_model()
