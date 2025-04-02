import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import einops
import os
import datetime
from trainAttn_fakedata import SelfAttentionLightning
from data_generation import prepare_data
from sae_fakedata import SparseAutoEncoder, calculate_loss

def train_sae_on_layer(model, layer_name, layer_output, save_dir):
    """
    Train a sparse autoencoder on a specific layer's output
    """
    # Create save directory for this layer's SAE
    layer_save_dir = os.path.join(save_dir, f"sae_{layer_name}")
    os.makedirs(layer_save_dir, exist_ok=True)
    
    # Handle tuple output from attention layer
    if isinstance(layer_output, tuple):
        # For attention layer, we'll train on both output and attention weights
        output, attn_weights = layer_output
        # Train on output
        print(f"\nTraining SAE on {layer_name}_output...")
        train_single_output(model, f"{layer_name}_output", output, layer_save_dir)
        # Train on attention weights
        print(f"\nTraining SAE on {layer_name}_weights...")
        train_single_output(model, f"{layer_name}_weights", attn_weights, layer_save_dir)
        return None  # No single autoencoder to return
    else:
        return train_single_output(model, layer_name, layer_output, layer_save_dir)

def train_single_output(model, name, tensor_output, save_dir):
    """
    Train a sparse autoencoder on a single tensor output
    """
    # Get the activation dimensions
    # If the tensor is 3D (batch, sequence, features), flatten sequence and features
    if len(tensor_output.shape) == 3:
        batch_size, seq_len, feat_dim = tensor_output.shape
        tensor_output = tensor_output.reshape(batch_size, seq_len * feat_dim)
        activation_dim = seq_len * feat_dim
    else:
        activation_dim = tensor_output.shape[-1]
    
    dict_size = activation_dim * 10  # Standard dictionary size multiplier
    
    # Initialize SAE
    autoencoder = SparseAutoEncoder(activation_dim, dict_size)
    
    # Training parameters
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 10
    l1_coefficient = 1e-5
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(tensor_output)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (batch_data,) in dataloader:
            optimizer.zero_grad()
            loss = calculate_loss(autoencoder, batch_data, l1_coefficient)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_data.size(0)
        epoch_loss /= len(dataset)
        print(f'{name} - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Save the trained SAE
    torch.save(autoencoder.state_dict(), os.path.join(save_dir, f'sae_{name}.pt'))
    
    # Save the shape information for later reconstruction
    shape_info = {
        'original_shape': list(tensor_output.shape),
        'flattened_dim': activation_dim
    }
    torch.save(shape_info, os.path.join(save_dir, f'shape_info_{name}.pt'))
    
    return autoencoder

def main():
    # Load the trained attention model
    model_save_path = "/media/dan/Data/git/ubiquitous-spork/prediction/src/models/quantized_attention_model_20250401_114748/final_model.ckpt"  # Update this path
    model = SelfAttentionLightning.load_from_checkpoint(model_save_path)
    model.eval()
    
    # Prepare data
    train_dataset, val_dataset, n_features = prepare_data()
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=min(63, os.cpu_count() or 1),
        pin_memory=True
    )
    
    # Create save directory for all SAEs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/media/dan/Data/git/ubiquitous-spork/prediction/src/models/sae_attention_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect activations from each layer
    layer_outputs = {}
    
    # Hook function to collect layer outputs
    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output
        return hook
    
    # Register hooks for the layers we want to analyze
    model.proj_keys.register_forward_hook(hook_fn('proj_keys'))
    model.proj_queries.register_forward_hook(hook_fn('proj_queries'))
    model.proj_values.register_forward_hook(hook_fn('proj_values'))
    model.attn.register_forward_hook(hook_fn('attn'))
    model.ln.register_forward_hook(hook_fn('ln'))
    model.fc1.register_forward_hook(hook_fn('fc1'))
    model.fc2.register_forward_hook(hook_fn('fc2'))
    
    # Forward pass to collect activations
    with torch.no_grad():
        for batch in train_loader:
            features, _ = batch
            model(features)
            break  # We only need one batch for now
    
    # Train SAE on each layer's output
    trained_saes = {}
    for layer_name, layer_output in layer_outputs.items():
        print(f"\nTraining SAE on {layer_name} layer...")
        trained_saes[layer_name] = train_sae_on_layer(model, layer_name, layer_output, save_dir)
    
    print(f"\nAll SAEs have been trained and saved to: {save_dir}")
    return trained_saes

if __name__ == "__main__":
    trained_saes = main() 