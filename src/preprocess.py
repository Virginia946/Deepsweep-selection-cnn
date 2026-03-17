# -----------------------------
# Combine and label data
# -----------------------------
X = np.concatenate([neutral, selection], axis=0)

# 0 = neutral, 1 = selection for supervised learning 
y = np.concatenate([
    np.zeros(len(neutral)),
    np.ones(len(selection))
])

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Add channel dimension for CNN
# From (samples, 40, 100) to (samples, 1, 40, 100)
X_tensor = X_tensor.unsqueeze(1)
