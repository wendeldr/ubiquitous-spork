import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

def attention_fakedata(quantize=False,quantization_decimals=2):
    rng = np.random.default_rng(42)
    ratio = 11.0

    # Generate synthetic data with 
    n_samples = 100000
    n_pos = int(n_samples / (ratio + 1))  # Calculate number of positive samples
    n_neg = n_samples - n_pos  # Remaining are negative samples

    # Create labels array with the correct ratio
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    # Shuffle the labels
    rng.shuffle(labels)

    # Create a DataFrame with the synthetic data
    raw_df = pd.DataFrame({'label': labels})

    # Using boolean indexing:
    mask0 = raw_df['label'] == 0
    mask1 = raw_df['label'] == 1

    raw_df.loc[mask0, 'f1'] = rng.normal(6, 0.3, size=mask0.sum())
    raw_df.loc[mask1, 'f1'] = rng.gamma(1, .11, size=mask1.sum())

    raw_df['f2'] = rng.normal(10, 5, size=n_samples)

    if quantize:
        raw_df['f1'] = np.round(raw_df['f1'], quantization_decimals)
        raw_df['f2'] = np.round(raw_df['f2'], quantization_decimals)

    neg, pos = np.bincount(raw_df['label'])
    total = neg + pos
    # print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    #     total, pos, 100 * pos / total))

    initial_bias = np.log([pos/neg])

    # print('data shape (+1 for label)')
    print(raw_df.shape)
    return raw_df, initial_bias

def prepare_data(quantize=False,quantization_decimals=2):
    raw_df, initial_bias = attention_fakedata(quantize,quantization_decimals)
    
    # Split dataset
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Prepare labels and features
    train_labels = np.array(train_df.pop('label')).reshape(-1, 1)
    val_labels = np.array(val_df.pop('label')).reshape(-1, 1)
    test_labels = np.array(test_df.pop('label')).reshape(-1, 1)
    
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    
    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    
    # Convert to tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    
    return train_dataset, val_dataset, train_features.shape[1] 