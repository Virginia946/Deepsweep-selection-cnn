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


