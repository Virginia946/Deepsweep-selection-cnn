def create_selection_dataset(n_samples, n_individuals, n_snps):
    windows = [
        simulate_selection_window(n_individuals, n_snps)
        for _ in range(n_samples)
    ]
    return np.stack(windows)


def create_neutral_dataset(n_samples, n_individuals, n_snps):
    windows = [
        simulate_neutral_window(n_individuals, n_snps)
        for _ in range(n_samples)
    ]
    return np.stack(windows)


selection = create_selection_dataset(500, n_individuals=40, n_snps=100)
neutral = create_neutral_dataset(500, n_individuals=40, n_snps=100)

print("Neutral shape:", neutral.shape)
print("Selection shape:", selection.shape)


class SNPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = SNPDataset(X_tensor, y_tensor)

