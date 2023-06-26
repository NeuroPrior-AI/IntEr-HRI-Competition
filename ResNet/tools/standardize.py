import numpy as np

def standardize(X, X_val):
    # Standardize data
    # Note that X has shape (num_samples, num_features, sequence_length)
    mean = np.mean(X, axis=(0, 2), keepdims=True)  # Calculate mean across samples and sequence length
    std = np.std(X, axis=(0, 2), keepdims=True)  # Calculate standard deviation across samples and sequence length
    # Perform standardization (Z-score normalization)
    X = (X - mean) / (std + 1e-7)
    # Standardize validation data
    X_val = (X_val - mean) / (std + 1e-7)

    return X, X_val
