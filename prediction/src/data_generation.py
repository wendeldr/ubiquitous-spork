import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

def attention_fakedata(n_features,quantize=False,quantization_decimals=2,n_samples=100000,ratio=11.0,seed=42):
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate synthetic data with 
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


    distributions = {
    'beta': {'obj': rng.beta, 'params': [{'a': 2, 'b': 5}, {'a': 5, 'b': 2}]},
    # 'binomial': {'obj': rng.binomial, 'params': {'n': 1, 'p': 0.5}},
    'chisquare': {'obj': rng.chisquare, 'params': [{'df': 1}, {'df': 10}]},
    # 'dirichlet': {'obj': rng.dirichlet, 'params': {'alpha': [1, 1, 1]}},
    'exponential': {'obj': rng.exponential, 'params': [{'scale': 1}, {'scale': 10}]},
    'f': {'obj': rng.f, 'params': [{'dfnum': 4, 'dfden': 10}, {'dfnum': 10, 'dfden': 4}]},
    'gamma': {'obj': rng.gamma, 'params': [{'shape': 1, 'scale': 1}, {'shape': 10, 'scale': 1}]},
    'geometric': {'obj': rng.geometric, 'params': [{'p': 0.5}, {'p': 0.1}]},
    'gumbel': {'obj': rng.gumbel, 'params': [{'loc': 0, 'scale': 1}, {'loc': 10, 'scale': 1}]},
    # 'hypergeometric': {'obj': rng.hypergeometric, 'params': {'ngood': 1, 'nbad': 1, 'nsample': 1}},
    'laplace': {'obj': rng.laplace, 'params': [{'loc': 0, 'scale': 1}, {'loc': 5, 'scale': 1}]},
    'logistic': {'obj': rng.logistic, 'params': [{'loc': 0, 'scale': 1}, {'loc': 20, 'scale': 1}]},
    'lognormal': {'obj': rng.lognormal, 'params': [{'mean': 0, 'sigma': 1}, {'mean': 5, 'sigma': 1.1}]},
    'logseries': {'obj': rng.logseries, 'params': [{'p': 0.5}, {'p': 0.1}]},
    # 'multinomial': {'obj': rng.multinomial, 'params': {'n': 10, 'pvals': [0.5, 0.5]}},
    # "multivariate_hypergeometric": {'obj': rng.multivariate_hypergeometric, 'params': {'nsample': 1, 'colors': [10, 10]}},
    # "multivariate_normal": {'obj': rng.multivariate_normal, 'params': {'mean': [0, 0], 'cov': [[1, 0], [0, 1]]}},
    "negative_binomial": {'obj': rng.negative_binomial, 'params': [{'n': 1, 'p': 0.5}, {'n': 10, 'p': 0.1}]},
    "noncentral_chisquare": {'obj': rng.noncentral_chisquare, 'params': [{'df': 1, 'nonc': 0}, {'df': 10, 'nonc': 0}]},
    "noncentral_f": {'obj': rng.noncentral_f, 'params': [{'dfnum': 4, 'dfden': 10, 'nonc': 0}, {'dfnum': 10, 'dfden': 4, 'nonc': 0}]},
    "normal": {'obj': rng.normal, 'params': [{'loc': 0, 'scale': 1}, {'loc': 15, 'scale': 1}]},
    "pareto": {'obj': rng.pareto, 'params': [{'a': .5}, {'a': 10}]},
    "poisson": {'obj': rng.poisson, 'params': [{'lam': 1}, {'lam': 10}]},
    "power": {'obj': rng.power, 'params': [{'a': 5}, {'a': 10}]},
    "rayleigh": {'obj': rng.rayleigh, 'params': [{'scale': 1}, {'scale': 10}]},
    "standard_cauchy": {'obj': rng.standard_cauchy, 'params': [{}, {}]},
    "standard_exponential": {'obj': rng.standard_exponential, 'params': [{}, {}]},
    "standard_gamma": {'obj': rng.standard_gamma, 'params': [{'shape': 1}, {'shape': 10}]},
    "standard_normal": {'obj': rng.standard_normal, 'params': [{}, {}]},
    "standard_t": {'obj': rng.standard_t, 'params': [{'df': 10}, {'df': 10}]},
    "triangular": {'obj': rng.triangular, 'params': [{'left': 0, 'mode': 0.5, 'right': 1}, {'left': 0, 'mode': 0.5, 'right': 1}]},
    "uniform": {'obj': rng.uniform, 'params': [{'low': 0, 'high': 1}, {'low': 0, 'high': 1}]},
    "vonmises": {'obj': rng.vonmises, 'params': [{'mu': 0, 'kappa': 1}, {'mu': 0, 'kappa': 1}]},
    "wald": {'obj': rng.wald, 'params': [{'mean': 1, 'scale': 1}, {'mean': 1, 'scale': 1}]},
    "weibull": {'obj': rng.weibull, 'params': [{'a': 1}, {'a': 10}]},
    "zipf": {'obj': rng.zipf, 'params': [{'a': 2}, {'a': 2}]},
    }


    clear_sep =[{'obj': rng.normal, 'params': {'loc': 0, 'scale': 1}},
                {'obj': rng.normal, 'params': {'loc': 100, 'scale': 1}}]
    no_sep = [{'obj': rng.uniform, 'params': {'low': 0, 'high': 1}},
              {'obj': rng.uniform, 'params': {'low': 0, 'high': 1}}]

    for f in range(n_features):
        if f == 0:
            dist = clear_sep
        else:
            dist = no_sep
        raw_df.loc[mask0, f'f{f}'] = dist[0]['obj'](**dist[0]['params'], size=mask0.sum())
        raw_df.loc[mask1, f'f{f}'] = dist[1]['obj'](**dist[1]['params'], size=mask1.sum())


    if quantize:
        for f in range(n_features):
            raw_df[f'f{f}'] = np.round(raw_df[f'f{f}'], quantization_decimals)

    # neg, pos = np.bincount(raw_df['label'])
    # total = neg + pos
    # # print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    # #     total, pos, 100 * pos / total))

    # initial_bias = np.log([pos/neg])

    # print('data shape (+1 for label)')
    print(raw_df.shape)
    return raw_df

def prepare_data(n_features,quantize=False,quantization_decimals=2,n_samples=100000,ratio=11.0,seed=42):
    raw_df = attention_fakedata(n_features,quantize,quantization_decimals,n_samples,ratio,seed)
    
    # Split dataset
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df['label'])
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['label'])

    
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
    
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    return train_dataset, val_dataset, test_dataset