"""Generate validated trajectories and store them in an HDF5 dataset."""

import h5py
import numpy as np

from dataset_generator import SEQUENCE_TIME, STEP, generate_valid_sequence

n_trajectories = 1000
n_steps = round(SEQUENCE_TIME / STEP) + 1

with h5py.File("three_body.h5", "w") as f:
    positions = f.create_dataset(
        "positions",
        shape=(n_trajectories, n_steps, 3, 2),
        dtype="float32",
        chunks=(1, n_steps, 3, 2),
        compression="lzf",
    )

    velocities = f.create_dataset(
        "velocities",
        shape=(n_trajectories, n_steps, 3, 2),
        dtype="float32",
        chunks=(1, n_steps, 3, 2),
        compression="lzf",
    )

    rng = np.random.default_rng()
    for i in range(n_trajectories):
        print(i)
        if i % 100 == 0:
            print(f"Generating {i}/{n_trajectories}")

        x, y, vx, vy = generate_valid_sequence(rng)

        positions[i] = np.stack((x, y), axis=-1)
        velocities[i] = np.stack((vx, vy), axis=-1)
