"""Throwaway script: plot true vs. predicted trajectories for a trained model.

Examples:
    python src/plot_predictions.py mlp
    python src/plot_predictions.py resnet --steps 1500 --trajectory-index 3
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from main import create_model
from utilts import plot_trajectories

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true and model-predicted three-body trajectories."
    )
    parser.add_argument(
        "model",
        choices=("mlp", "resnet", "resnetv2"),
        help="Model to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: checkpoints/<model>/model.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="HDF5 dataset (default: path stored in the checkpoint).",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="First dataset trajectory to plot.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="Number of consecutive trajectories to plot.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Rollout length (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: outputs/predictions).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, ... (default: auto).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the current directory or the repo root."""
    if path.is_absolute() or path.exists():
        return path

    repo_relative = ROOT / path
    return repo_relative if repo_relative.exists() else path


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_trajectory(dataset_path: Path, index: int, steps: int) -> np.ndarray:
    """Load one trajectory from the HDF5 file in the training state layout."""
    with h5py.File(dataset_path, "r") as file:
        total_points = file["positions"].shape[1]

        if steps + 1 > total_points:
            raise ValueError(
                f"Requested {steps} steps but trajectory only has "
                f"{total_points - 1} transitions."
            )

        positions = file["positions"][index, : steps + 1].reshape(steps + 1, 6)
        velocities = file["velocities"][index, : steps + 1].reshape(steps + 1, 6)

    return np.concatenate((positions, velocities), axis=-1)


def load_model(checkpoint_path: Path, model_type: str, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_type", model_type) != model_type:
        raise ValueError(
            f"Checkpoint is for '{checkpoint['model_type']}', not '{model_type}'."
        )

    checkpoint_config = checkpoint.get("config") or {}
    model_cfg = dict(checkpoint_config.get("model", {}))
    model_cfg.setdefault("type", model_type)
    model_cfg.setdefault("dt", 1e-4)

    model = create_model(OmegaConf.create({"model": model_cfg}), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_trajectory(model, states: np.ndarray, steps: int, device: torch.device):
    initial_state = torch.as_tensor(states[:1], dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted = model.predict(initial_state, steps=steps)

    return predicted.squeeze(0).cpu().numpy()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    checkpoint_path = resolve_path(
        args.checkpoint or ROOT / "checkpoints" / args.model / "model.pt"
    )
    model, checkpoint = load_model(checkpoint_path, args.model, device)

    checkpoint_config = checkpoint.get("config") or {}
    dataset_path = resolve_path(
        args.dataset
        or Path(checkpoint_config.get("dataset", {}).get("path", "dataset_1k_v1.h5"))
    )
    output_dir = args.output_dir or ROOT / "outputs" / "predictions"

    with h5py.File(dataset_path, "r") as file:
        available = file["positions"].shape[0]

    first = args.trajectory_index
    last = min(first + args.num_trajectories, available)

    if first >= available:
        raise ValueError(
            f"Trajectory index {first} out of range for {available} trajectories."
        )

    print(f"Model: {args.model} ({checkpoint_path})")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Rollout: {args.steps} steps")

    for index in range(first, last):
        true_states = load_trajectory(dataset_path, index, args.steps)
        predicted_states = predict_trajectory(model, true_states, args.steps, device)

        save_dir = output_dir / f"trajectory_{index:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_trajectories(
            true_states,
            predicted_states,
            save_dir,
            model_type=args.model,
            steps=args.steps,
        )

        position_error = np.linalg.norm(
            predicted_states[:, :6].reshape(-1, 3, 2)
            - true_states[:, :6].reshape(-1, 3, 2),
            axis=2,
        )
        figure_path = save_dir / f"trajectories_{args.model}_{args.steps}.png"
        print(
            f"Trajectory {index}: {figure_path} "
            f"(max position error {position_error.max():.3e})"
        )


if __name__ == "__main__":
    main()
