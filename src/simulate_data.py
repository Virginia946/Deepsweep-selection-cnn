

import numpy as np
#simulation of neutral and selective signatures to train the network

import numpy as np


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

    start = np.random.randint(0, n_snps - block_size + 1)
    stop = start + block_size

    for snp in range(start, stop):
        dominant = np.random.choice([0, 2])
        n_selected = int(0.8 * n_individuals)
        selected_inds = np.random.choice(n_individuals, size=n_selected, replace=False)
        window[selected_inds, snp] = dominant

    return window


neutral=simulate_neutral_window(40,100)

selection=simulate_selection_window(40,100,20)


#create the dataset 1000 (500 neutral/500 selective )


#def create_neutral_dataset()




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

selection=(create_selection_dataset(500,n_individuals=40,n_snps=100))
neutral=(create_neutral_dataset(n_samples=500, n_individuals=40, n_snps=100))
print(neutral)







