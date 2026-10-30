"""Load trajectory data and split it into PyTorch data loaders."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(
    path: str,
    number_trajectories: int = 50,
    points_trajectory: int = 50,
    batchsize: int = 32,
    training_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> torch.Tensor:
    """
    Prepare current-state and next-state pairs for neural-network training.

    Each trajectory is flattened into a 12-value state consisting of the
    positions and velocities of the three bodies. Consecutive states form an
    input-target pair, and those pairs are split reproducibly into training,
    validation, and test loaders.

    Args:
        path: Path to an HDF5 file with ``positions`` and ``velocities``
            datasets.
        number_trajectories: Maximum number of trajectories to load.
        points_trajectory: Maximum number of time points to load per
            trajectory.
        batchsize: Number of state pairs in each loader batch.
        training_fraction: Fraction of samples assigned to training.
        validation_fraction: Fraction of samples assigned to validation.
        seed: Seed used for the reproducible random split.

    Returns:
        Training, validation, and test ``DataLoader`` objects.
    """
    filepath = Path(path)

    with h5py.File(filepath, "r") as file:
        print(file.keys())
        print(file["positions"].shape)
        positions = file["positions"][:number_trajectories, :points_trajectory]
        velocities = file["velocities"][:number_trajectories, :points_trajectory]

    print(f"positions shape: {positions.shape}")
    print(f"velocities shape: {velocities.shape}")

    positions = positions.reshape(*positions.shape[:2], 6)
    velocities = velocities.reshape(*velocities.shape[:2], 6)
    print(f"positions shape: {positions.shape}")
    print(f"velocities shape: {velocities.shape}")

    inputs = np.concatenate([positions, velocities], axis=-1)

    x = inputs[:, :-1, :]
    y = inputs[:, 1:, :]
    print(x.shape)
    print(y.shape)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x, y)
    samples_train = int(training_fraction * len(dataset))
    samples_val = int(validation_fraction * len(dataset))
    samples_eval = len(dataset) - samples_train - samples_val

    print(
        f"size training dataset: {samples_train}, size val dataset: {samples_val}, size eval dataset: {samples_eval}"
    )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [samples_train, samples_val, samples_eval], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batchsize)
    val_loader = DataLoader(val_dataset, batch_size=batchsize)
    test_loader = DataLoader(test_dataset, batch_size=batchsize)

    return train_loader, val_loader, test_loader
