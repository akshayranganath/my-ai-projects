"""
Test script for GGUF model conversion and inference.
"""
import torch
from torchvision import datasets, transforms
from convert.to_gguf import convert_to_gguf, inference_gguf
import matplotlib.pyplot as plt
import numpy as np
import os
def test_gguf_model():
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Convert model to GGUF
    print("Current working directory:", os.getcwd())
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use paths relative to src directory
    model_path = os.path.join(script_dir, 'models', 'mnist_best.pt')
    gguf_path = os.path.join(script_dir, 'models', 'mnist.gguf')
    
    # Ensure the models directory exists
    os.makedirs(os.path.join(script_dir, 'models'), exist_ok=True)
    convert_to_gguf(model_path, gguf_path)
    print(f"Model converted and saved to {gguf_path}")
    
    # Test on a few examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, (data, target) in enumerate(test_loader):
        if i >= 10:  # Test first 10 images
            break
            
        # Get prediction
        pred = inference_gguf(gguf_path, data)
        predicted_class = np.argmax(pred)
        
        # Plot image and prediction
        axes[i].imshow(data[0, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'True: {target.item()}\nPred: {predicted_class}')
    
    plt.tight_layout()
    # Save plot in the plots directory
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'gguf_test_predictions.png'))
    plt.close()
    print("Test predictions saved to plots/gguf_test_predictions.png")

if __name__ == '__main__':
    test_gguf_model()
