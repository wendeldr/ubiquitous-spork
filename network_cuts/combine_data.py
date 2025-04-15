import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def combine_data(data):
    # Combine mixed groups
    mix = np.concatenate([data['non_soz'], data['soz_non']])
    
    # Create arrays for each group
    mix_data = mix
    non_data = data['non_non']
    soz_data = data['soz_soz']
    
    # Create labels for each group
    mix_labels = ['mix'] * len(mix_data)
    non_labels = ['non'] * len(non_data)
    soz_labels = ['soz'] * len(soz_data)
    
    # Combine all data and labels
    all_data = np.concatenate([mix_data, non_data, soz_data])
    all_labels = mix_labels + non_labels + soz_labels
    
    # Create DataFrame
    df = pd.DataFrame({
        'value': all_data,
        'group': all_labels
    })
    
    return df

def create_legend_plot(output_path=None, orientation='horizontal'):
    """
    Create a legend plot with specified orientation.
    
    Parameters:
    -----------
    output_path : str, optional
        Path to save the legend plot. If None, the plot is displayed.
    orientation : str, optional
        'horizontal' or 'vertical' for the legend layout.
    """
    # Set figure size based on orientation
    if orientation == 'horizontal':
        plt.figure(figsize=(3, 0.5))
        ncol = 3
    else:  # vertical
        plt.figure(figsize=(1, 1.5))
        ncol = 1
    
    # Create dummy data points for each group
    x = [0, 1, 2]
    y = [0, 0, 0]
    
    # Plot points with the specified colors
    plt.scatter(x, y, c=['black', '#D41159', '#1A85FF'], s=100)
    
    # Create legend with custom labels
    plt.legend(['Non-EZ', 'EZ', 'Non->EZ'], 
               loc='center',
               ncol=ncol,
               frameon=False)
    
    # Remove axes and background
    plt.axis('off')
    
    # Adjust layout to show only the legend
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        # Add orientation to filename if not already present
        if not output_path.endswith('.png'):
            output_path += '.png'
        base_path = output_path.rsplit('.', 1)[0]
        output_path = f"{base_path}_{orientation}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# Example usage:
# data = {'non_non': array1, 'non_soz': array2, 'soz_non': array3, 'soz_soz': array4}
# df = combine_data(data)
# create_legend_plot('path/to/save/legend', orientation='horizontal')
# create_legend_plot('path/to/save/legend', orientation='vertical') 