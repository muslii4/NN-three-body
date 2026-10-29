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
    This function prepares the dataset to be the input for a neural network.
    We construct pairs based on the concept <current state, next state>.

    Args:
    path (str): The path of the file
    number_trajectories (int): The number of trajectories from the entire dataset that are used for training/testing the model.
    points_trajectory (int): The number of discrete points in each trajectory chosen for training/testing the model.
    batchsize (int): Preferred batchsize.
    training_fraction (int): The percentage of the dataset that is used for training.
    validation_fraction (int): The percentage of the dataset that is used for validation.
    seed (int): To ensure the robustness of the results, the seed is fixed during the experiments.

    Returns:
    train_loader, val_loader, test_loader (torch.Tensor, torch.Tensor, torch.Tensor): Loaded data for each dataset.
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
