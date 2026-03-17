# import relevant modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# Simulate neutral and selective windows
# -----------------------------
def simulate_neutral_window(n_individuals: int, n_snps: int) -> np.ndarray:
    allele_frequency = np.random.uniform(0.1, 0.9, size=n_snps)
    window = np.random.binomial(2, allele_frequency, size=(n_individuals, n_snps))
    return window


def simulate_selection_window(
    n_individuals: int,
    n_snps: int,
    block_size: int = 20
) -> np.ndarray:
    window = simulate_neutral_window(n_individuals, n_snps)

    # 0 = smallest possible start index
    # n_snps - block_size = last valid start
    # +1 is needed because randint excludes the upper bound
    start = np.random.randint(0, n_snps - block_size + 1)
    stop = start + block_size

    for snp in range(start, stop):
        dominant = np.random.choice([0, 2])
        n_selected = int(0.8 * n_individuals)
        selected_inds = np.random.choice(n_individuals, size=n_selected, replace=False)
        window[selected_inds, snp] = dominant

    return window


# -----------------------------
# Create datasets
# -----------------------------
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

# -----------------------------
# Combine and label data
# -----------------------------
X = np.concatenate([neutral, selection], axis=0)

# 0 = neutral, 1 = selection
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

# -----------------------------
# Dataset and DataLoader
# -----------------------------
class SNPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = SNPDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Define CNN model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Input: (1, 40, 100)
        # After conv1 + pool: (16, 20, 50)
        # After conv2 + pool: (32, 10, 25)
        self.fc1 = nn.Linear(32 * 10 * 25, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (16, 20, 50)
        x = self.pool(F.relu(self.conv2(x)))   # (32, 10, 25)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Set up training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # shape: (batch_size, 1)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# -----------------------------
# Evaluation on test set
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")


model.eval()

new_window = simulate_selection_window(40, 100)   # or your real data window
new_tensor = torch.tensor(new_window, dtype=torch.float32)

# add channel and batch dimensions
new_tensor = new_tensor.unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(new_tensor)
    prob = torch.sigmoid(output)
    pred = (prob > 0.5).float()

print("Probability of selection:", prob.item())
print("Predicted class:", int(pred.item()))