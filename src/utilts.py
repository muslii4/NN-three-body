import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_history(train_losses, validation_losses, save_dir, model_type):
    """Save training and validation loss curves for one run."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training loss")
    ax.plot(epochs, validation_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_type.upper()} training and validation loss")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"loss_curve_{model_type}.png"),
        dpi=200,
    )
    plt.close(fig)


def calculate_energy(states, device):
    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        velocities = states[..., 6:].reshape(*states.shape[:-1], 3, 2)
        positions = states[..., :6].reshape(*states.shape[:-1], 3, 2)
        kinetic = 0.5 * torch.sum(velocities**2, dim=(-1, -2))
        distances = torch.linalg.vector_norm(
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            dim=-1,
        )
        pair_distances = torch.stack(
            (distances[..., 0, 1], distances[..., 0, 2], distances[..., 1, 2]),
            dim=-1,
        ).clamp_min(1e-8)
        potential = -torch.sum(1.0 / pair_distances, dim=-1)
        total = kinetic + potential

    return (
        kinetic.cpu().numpy(),
        potential.cpu().numpy(),
        total.cpu().numpy(),
    )


def calculate_vector_fields(model, states, device):
    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        true_field = model.true_vector_field(states)
        learned_field = model.learned_vector_field(states)

    return (
        true_field.cpu().numpy(),
        learned_field.cpu().numpy(),
    )


def plot_trajectories(
    true_states, predicted_states, save_dir, model_type="resnet", steps=1
):

    true_positions = true_states[:, :6].reshape(-1, 3, 2)
    predicted_positions = predicted_states[:, :6].reshape(-1, 3, 2)

    fig, ax = plt.subplots(figsize=(9, 8))

    for i in range(3):
        ax.plot(
            true_positions[:, i, 0],
            true_positions[:, i, 1],
            label=f"Body {i + 1} - True",
        )

        ax.plot(
            predicted_positions[:, i, 0],
            predicted_positions[:, i, 1],
            "--",
            label=f"Body {i + 1} - Predicted",
        )

        ax.scatter(
            true_positions[0, i, 0],
            true_positions[0, i, 1],
            marker="o",
            s=35,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Three-body trajectory comparison")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"trajectories_{model_type}_{steps}.png"), dpi=200
    )
    plt.close(fig)


def plot_position_error(
    true_states, predicted_states, dt, save_dir, model_type="resnet", steps=1
):

    true_positions = true_states[:, :6].reshape(-1, 3, 2)
    predicted_positions = predicted_states[:, :6].reshape(-1, 3, 2)

    # Euclidean distance between predicted and true positions
    errors = np.linalg.norm(
        predicted_positions - true_positions,
        axis=2,
    )

    time = np.arange(len(true_states)) * dt

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(3):
        ax.plot(
            time,
            errors[:, i],
            label=f"Body {i + 1}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Position error")
    ax.set_title("Position prediction error over time")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"position_error_{model_type}_{steps}.png"), dpi=200
    )
    plt.close(fig)


def plot_energy(
    model,
    true_states,
    predicted_states,
    dt,
    device,
    save_dir,
    model_type="resnet",
    steps=1,
):

    true_T, true_V, true_E = calculate_energy(true_states, device)

    pred_T, pred_V, pred_E = calculate_energy(predicted_states, device)

    time = np.arange(len(true_states)) * dt

    energy_data = [
        (true_T, pred_T, "Kinetic energy difference", "ΔT"),
        (true_V, pred_V, "Potential energy difference", "ΔV"),
        (true_E, pred_E, "Total energy difference", "ΔE"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for ax, (true_energy, pred_energy, title, ylabel) in zip(axes, energy_data):
        ax.plot(
            time,
            true_energy - true_energy[0],
            label="Ground truth",
        )

        ax.plot(
            time,
            pred_energy - pred_energy[0],
            "--",
            label="Prediction",
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"energy_differences_{model_type}_{steps}.png"),
        dpi=200,
    )
    plt.close(fig)


def plot_vector_field(
    model, reference_state, device, save_dir, grid_size=20, model_type="resnet"
):
    model.eval()
    reference_state = np.asarray(reference_state, dtype=np.float32)

    # x-axis: x1
    # y-axis: vx1

    x_values = np.linspace(
        reference_state[0] - 1.0,
        reference_state[0] + 1.0,
        grid_size,
    )

    vx_values = np.linspace(
        reference_state[6] - 1.0,
        reference_state[6] + 1.0,
        grid_size,
    )

    X, VX = np.meshgrid(x_values, vx_values)

    states = np.repeat(
        reference_state[None, :],
        grid_size * grid_size,
        axis=0,
    )

    states[:, 0] = X.ravel()
    states[:, 6] = VX.ravel()

    true_field, learned_field = calculate_vector_fields(model, states, device)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fields = [
        (true_field, "True vector field"),
        (learned_field, "Learned vector field"),
    ]

    for ax, (field, title) in zip(axes, fields):
        # dx1/dt = vx1
        U = field[:, 0].reshape(X.shape)
        # dvx1/dt = ax1
        V = field[:, 6].reshape(X.shape)
        ax.quiver(
            X,
            VX,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=None,
        )

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$v_{x1}$")
        ax.set_title(title)
        ax.grid(True)

    fig.suptitle("True versus learned vector field")
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, f"vector_field_{model_type}.png"), dpi=200)
    plt.close(fig)


def numerical_diagnostics(
    model,
    true_states,
    predicted_states,
    device,
    model_type="resnet",
):

    true_states = np.asarray(true_states)
    predicted_states = np.asarray(predicted_states)

    true_pos = true_states[:, :6].reshape(-1, 3, 2)
    pred_pos = predicted_states[:, :6].reshape(-1, 3, 2)
    true_vel = true_states[:, 6:].reshape(-1, 3, 2)
    pred_vel = predicted_states[:, 6:].reshape(-1, 3, 2)

    position_difference = pred_pos - true_pos
    velocity_difference = pred_vel - true_vel

    position_mse = np.mean(position_difference**2)
    position_mae = np.mean(np.abs(position_difference))
    velocity_mse = np.mean(velocity_difference**2)
    velocity_mae = np.mean(np.abs(velocity_difference))

    position_errors = np.linalg.norm(
        position_difference,
        axis=2,
    )

    max_position_error = np.max(position_errors)

    metrics = {
        "position_mse": position_mse,
        "position_mae": position_mae,
        "velocity_mse": velocity_mse,
        "velocity_mae": velocity_mae,
        "max_position_error": max_position_error,
    }

    print("\n========== NUMERICAL DIAGNOSTICS ==========")
    print("\nPrediction accuracy:")
    print(f"Position MSE:             {position_mse}")
    print(f"Position MAE:             {position_mae}")
    print(f"Velocity MSE:             {velocity_mse}")
    print(f"Velocity MAE:             {velocity_mae}")
    print(f"Maximum position error:   {max_position_error}")

    if model_type in ("resnet", "mlp"):
        _, _, predicted_energy = calculate_energy(predicted_states, device)
        true_T, true_V, _ = calculate_energy(true_states[:1], device)
        energy_scale = abs(true_T[0]) + abs(true_V[0]) + 1e-8
        relative_energy_drift = (
            np.abs(predicted_energy - predicted_energy[0]) / energy_scale
        )
        max_relative_energy_drift = np.max(relative_energy_drift)
        metrics["max_relative_energy_drift"] = max_relative_energy_drift
        print("\nPhysical consistency:")
        print(f"Maximum relative energy drift:{max_relative_energy_drift}")
    print("===========================================\n")
    return metrics


def run_diagnostics(
    model,
    true_states,
    predicted_states,
    dt,
    device,
    save_dir="outputs/diagnostics",
    model_type="resnet",
    steps=1,
):
    os.makedirs(save_dir, exist_ok=True)

    true_states = np.asarray(true_states)
    predicted_states = np.asarray(predicted_states)

    if model_type == "resnet":
        plot_trajectories(
            true_states, predicted_states, save_dir, model_type=model_type, steps=steps
        )

        plot_position_error(
            true_states,
            predicted_states,
            dt,
            save_dir,
            model_type=model_type,
            steps=steps,
        )

        plot_energy(
            model,
            true_states,
            predicted_states,
            dt,
            device,
            save_dir,
            model_type=model_type,
            steps=steps,
        )

        reference_state = true_states[len(true_states) // 2]

        plot_vector_field(
            model, reference_state, device, save_dir, model_type=model_type
        )

    elif model_type == "mlp":
        plot_trajectories(
            true_states,
            predicted_states,
            save_dir,
            model_type=model_type,
            steps=steps,
        )

        plot_position_error(
            true_states,
            predicted_states,
            dt,
            save_dir,
            model_type=model_type,
            steps=steps,
        )

        plot_energy(
            model,
            true_states,
            predicted_states,
            dt,
            device,
            save_dir,
            model_type=model_type,
            steps=steps,
        )

    # Numerical diagnostics
    metrics = numerical_diagnostics(
        model,
        true_states,
        predicted_states,
        device,
        model_type=model_type,
    )

    print(f"Diagnostic plots saved in: {save_dir}")
    return metrics


class EarlyStopper:
    def __init__(
        self,
        patience: int = 40,
        min_delta: float = 1e-5,
        max_overfit: float = 1.5,
        warmup_epochs: int = 15,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.max_overfit = max_overfit
        self.warmup_epochs = warmup_epochs

        self.best_loss = float("inf")
        self.counter = 0
        self.epoch = 0

    def __call__(
        self,
        val_loss: float,
        train_loss: float,
        overfit_max: bool = False,
    ) -> bool:
        self.epoch += 1

        # Ignore early overfitting for n steps
        if self.epoch <= self.warmup_epochs:
            self.best_loss = min(self.best_loss, val_loss)
            return False

        # Overfitting Check
        safe_train_loss = max(train_loss, 1e-8)
        overfit_ratio = val_loss / safe_train_loss

        if overfit_max and overfit_ratio > self.max_overfit:
            print(
                f"Overfitting detected: ratio={overfit_ratio:.4f} > {self.max_overfit}"
            )
            return True

        # Standard Early Stopping
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            print(f"Patience exceeded: no improvement for {self.patience} epochs.")
            return True

        return False
