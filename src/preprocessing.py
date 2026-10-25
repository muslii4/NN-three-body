from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(
    path: str, number_trajectories: int = 10, points_trajectory: int = 10
) -> torch.Tensor:
    """
    This function prepares the dataset to be the input for a neural network.
    We construct pairs based on the concept <current state, next state>.

    Args:
    path (str): The path of the file
    number_trajectories (int): The number of trajectories from the entire dataset that are used for training/testing the model.
    points_trajectory (int): The number of steps for discrete points in each trajectory chosen for training/testing the model.

    Returns:
    loader (torch.Tensor): The loaded data.
    """
    filepath = Path(path)

    with h5py.File(filepath, "r") as file:
        print(file.keys())
        print(file["positions"].shape)
        positions = file["positions"][:number_trajectories, ::points_trajectory]
        velocities = file["velocities"][:number_trajectories, ::points_trajectory]

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
    loader = DataLoader(dataset, batch_size=32)

    return loader
