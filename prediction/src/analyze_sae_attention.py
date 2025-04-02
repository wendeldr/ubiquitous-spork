import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sae_fakedata import SparseAutoEncoder
import os
from data_generation import prepare_data
from trainAttn_fakedata import SelfAttentionLightning

def load_sae_and_shape(sae_dir, name):
    """
    Load a trained SAE and its shape information
    """
    # Get the layer directory
    layer_dir = os.path.join(sae_dir, f'sae_{name}')
    
    # Load shape information
    shape_info = torch.load(os.path.join(layer_dir, f'shape_info_{name}.pt'), map_location='cpu')
    
    # Initialize SAE with the saved dimensions
    activation_dim = shape_info['flattened_dim']
    dict_size = activation_dim * 4  # Same multiplier as in training
    
    sae = SparseAutoEncoder(activation_dim, dict_size)
    sae.load_state_dict(torch.load(os.path.join(layer_dir, f'sae_{name}.pt'), map_location='cpu'))
    sae.eval()
    
    return sae, shape_info

def analyze_sae_weights(sae, name):
    """
    Analyze and visualize the SAE weights
    """
    # Get encoder and decoder weights
    encoder_weights = sae.encoder_DF.weight.data.numpy()
    decoder_weights = sae.decoder_FD.weight.data.numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot encoder weights
    sns.heatmap(encoder_weights, ax=ax1, cmap='RdBu_r', center=0)
    ax1.set_title(f'{name} - Encoder Weights')
    ax1.set_xlabel('Dictionary Size')
    ax1.set_ylabel('Input Dimension')
    
    # Plot decoder weights
    sns.heatmap(decoder_weights, ax=ax2, cmap='RdBu_r', center=0)
    ax2.set_title(f'{name} - Decoder Weights')
    ax2.set_xlabel('Output Dimension')
    ax2.set_ylabel('Dictionary Size')
    
    plt.tight_layout()
    return fig

def analyze_sae_reconstruction(sae, data, name):
    """
    Analyze reconstruction quality of the SAE
    """
    with torch.no_grad():
        reconstructed, encoded = sae.forward_pass(data)
        reconstruction_error = (reconstructed - data).pow(2).mean(dim=1)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot reconstruction error distribution
        sns.histplot(reconstruction_error.numpy(), ax=ax1, bins=50)
        ax1.set_title(f'{name} - Reconstruction Error Distribution')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Count')
        
        # Plot sparsity of encoded representation
        sparsity = (encoded > 0).float().mean(dim=1)
        sns.histplot(sparsity.numpy(), ax=ax2, bins=50)
        ax2.set_title(f'{name} - Sparsity Distribution')
        ax2.set_xlabel('Fraction of Active Neurons')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig

def main():
    # Load the trained attention model
    model_save_path = "/media/dan/Data/git/ubiquitous-spork/prediction/src/models/attention_model_20250331_173250/final_model.ckpt"
    model = SelfAttentionLightning.load_from_checkpoint(model_save_path, map_location='cpu')
    model.eval()
    
    # Load the SAE directory (update this path to your latest SAE training run)
    sae_dir = "/media/dan/Data/git/ubiquitous-spork/prediction/src/models/sae_attention_20250331_175038"
    
    # Prepare data
    train_dataset, val_dataset, n_features = prepare_data()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=min(63, os.cpu_count() or 1),
        pin_memory=False  # No need for pin_memory when using CPU
    )
    
    # Get a batch of data
    with torch.no_grad():
        features, _ = next(iter(train_loader))
    
    # List of layer names to analyze
    layer_names = [
        'proj_keys', 'proj_queries', 'proj_values',
        'attn_output', 'attn_weights', 'ln', 'fc1', 'fc2'
    ]
    
    # Create output directory for plots
    plots_dir = os.path.join(sae_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Analyze each layer's SAE
    for name in layer_names:
        print(f"\nAnalyzing {name}...")
        
        try:
            # Load SAE and shape info
            sae, shape_info = load_sae_and_shape(sae_dir, name)
            
            # Get layer activations
            with torch.no_grad():
                if name == 'attn_output' or name == 'attn_weights':
                    # Special handling for attention layer outputs
                    output, _ = model.attn(
                        model.proj_queries(features.unsqueeze(-1)),
                        model.proj_keys(features.unsqueeze(-1)),
                        model.proj_values(features.unsqueeze(-1))
                    )
                    if name == 'attn_weights':
                        _, output = model.attn(
                            model.proj_queries(features.unsqueeze(-1)),
                            model.proj_keys(features.unsqueeze(-1)),
                            model.proj_values(features.unsqueeze(-1))
                        )
                else:
                    # Forward pass through the model up to the desired layer
                    x = features.unsqueeze(-1)
                    if name == 'proj_keys':
                        output = model.proj_keys(x)
                    elif name == 'proj_queries':
                        output = model.proj_queries(x)
                    elif name == 'proj_values':
                        output = model.proj_values(x)
                    elif name == 'ln':
                        x = model.proj_queries(x)
                        x = model.attn(x, x, x)[0]
                        output = model.ln(x.reshape(x.size(0), -1))
                    elif name == 'fc1':
                        x = model.proj_queries(x)
                        x = model.attn(x, x, x)[0]
                        x = model.ln(x.reshape(x.size(0), -1))
                        output = model.fc1(x)
                    elif name == 'fc2':
                        x = model.proj_queries(x)
                        x = model.attn(x, x, x)[0]
                        x = model.ln(x.reshape(x.size(0), -1))
                        x = model.fc1(x)
                        output = model.fc2(x)
                
                # Reshape output if needed
                if len(output.shape) == 3:
                    output = output.reshape(output.size(0), -1)
            
            # Analyze weights
            weight_fig = analyze_sae_weights(sae, name)
            weight_fig.savefig(os.path.join(plots_dir, f'{name}_weights.png'))
            plt.close(weight_fig)
            
            # Analyze reconstruction
            recon_fig = analyze_sae_reconstruction(sae, output, name)
            recon_fig.savefig(os.path.join(plots_dir, f'{name}_reconstruction.png'))
            plt.close(recon_fig)
            
            # Print some statistics
            with torch.no_grad():
                reconstructed, encoded = sae.forward_pass(output)
                reconstruction_error = (reconstructed - output).pow(2).mean()
                sparsity = (encoded > 0).float().mean()
                print(f"Reconstruction Error: {reconstruction_error:.4f}")
                print(f"Sparsity: {sparsity:.4f}")
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 