"""Plot trajectories stored in a three-body HDF5 dataset."""

import h5py

from generate_data import plot_data, plot_data_animated

ANIMATE = False


def main() -> None:
    """Plot every trajectory from the configured HDF5 dataset."""
    with h5py.File("three_body.h5", "r") as file:
        positions = file["positions"]
        trajectory_indices = range(positions.shape[0])
        for trajectory_index in trajectory_indices:
            trajectory = positions[trajectory_index]
            x = trajectory[:, :, 0]
            y = trajectory[:, :, 1]

            if ANIMATE:
                plot_data_animated(x, y)
            else:
                plot_data(x, y)


if __name__ == "__main__":
    main()
