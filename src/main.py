"""Train a selected three-body model and evaluate test trajectories."""

import os
from copy import deepcopy

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from models.baseline import PhysicsResNet
from models.MLP import MLP
from models.resnetv2 import PhysicsResNetV2
from preprocessing import load_dataset
from training import train_one_epoch, validate_one_epoch
from utilts import EarlyStopper, plot_loss_history, run_diagnostics


def create_model(cfg: DictConfig, device: torch.device):
    """Create the requested model."""

    if cfg.model.type == "mlp":
        return MLP(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
        ).to(device)

    elif cfg.model.type == "resnet":
        return PhysicsResNet(
            dt=cfg.model.dt,
        ).to(device)

    elif cfg.model.type == "resnetv2":
        return PhysicsResNetV2(
            input_size=cfg.model.get("input_size", 12),
            hidden_size=cfg.model.get("hidden_size", 256),
            num_layers=cfg.model.get("num_layers", 5),
            G=cfg.model.get("G", 1.0),
            masses=tuple(cfg.model.get("masses", (1.0, 1.0, 1.0))),
            dt=cfg.model.dt,
            epsilon=cfg.model.get("epsilon", 1e-8),
            lambda_kinematics=cfg.model.get("lambda_kinematics", 0.1),
            lambda_dynamics=cfg.model.get("lambda_dynamics", 0.1),
            lambda_energy=cfg.model.get("lambda_energy", 0.01),
        ).to(device)

    raise ValueError(f"Unknown model type: {cfg.model.type}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_test_trajectories(test_loader, steps):
    """Extract all complete trajectories from the test dataset."""

    test_subset = test_loader.dataset
    trajectories = []

    for trajectory_index in test_subset.indices:
        inputs, targets = test_subset.dataset[trajectory_index]

        true_states = torch.cat((inputs[:1], targets), dim=0)

        if true_states.shape[0] < steps + 1:
            raise ValueError(f"Trajectory {trajectory_index} has insufficient states.")

        trajectories.append((trajectory_index, true_states[: steps + 1]))

    return trajectories


def predict_trajectory(model, true_states, device, steps):
    """Predict a complete trajectory from its initial state."""

    model.eval()
    initial_state = true_states[:1].to(device)

    with torch.no_grad():
        predicted = model.predict(initial_state, steps=steps)

    return predicted.squeeze(0).cpu().numpy()


def run_training(cfg: DictConfig):
    """Train the selected model and save the best checkpoint."""

    set_seed(cfg.training.get("seed", cfg.get("seed", 42)))

    type_model = cfg.model.type

    early_stopper = EarlyStopper(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        max_overfit=cfg.training.early_stopping.max_overfit,
    )

    training_loader, validation_loader, _ = load_dataset(
        cfg.dataset.path,
        number_trajectories=cfg.dataset.number_trajectories,
        points_trajectory=cfg.dataset.points_trajectory,
        batchsize=cfg.dataset.batch_size,
        flatten_transitions=cfg.dataset.get("flatten_transitions", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(cfg, device)

    print(f"Model: {cfg.model.type}")
    print(f"Device: {device}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_scheduler.factor,
        patience=cfg.training.lr_scheduler.patience,
        min_lr=cfg.training.lr_scheduler.min_lr,
    )

    loss_function = None
    validation_loss_function = None

    if cfg.model.type == "resnet":
        loss_function = model.loss_function
        validation_loss_function = model.loss_function

    if cfg.model.type == "resnetv2":
        loss_function = model.loss_function
        # Select checkpoints using the supervised loss only, so a physics
        # runaway cannot be chosen as the "best" model.
        validation_loss_function = model.data_loss

    best_validation_loss = float("inf")
    best_model_state = None

    train_losses = []
    validation_losses = []

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device,
            loss_function=loss_function,
            grad_clip=cfg.training.get("grad_clip", 1.0),
        )

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            loss_function=validation_loss_function,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.6e}, "
            f"validation={validation_loss:.6e}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if early_stopper(
            val_loss=validation_loss,
            train_loss=train_loss,
        ):
            print("Early stopping triggered.")
            break

    if best_model_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_model_state)

    plot_loss_history(
        train_losses,
        validation_losses,
        save_dir=os.getcwd(),
        model_type=type_model,
    )

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": cfg.model.type,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        os.path.join(cfg.training.output_dir, "model.pt"),
    )

    print("Best model saved successfully.")


def load_model(cfg: DictConfig, device: torch.device):
    """Load a trained model from a checkpoint."""

    checkpoint_path = os.path.join(
        cfg.training.output_dir,
        "model.pt",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint["model_type"] != cfg.model.type:
        raise ValueError("Checkpoint model type does not match configuration.")

    model = create_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def run_evaluation(cfg: DictConfig):
    """Evaluate all test trajectories and plot the first 10."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(cfg, device)

    _, _, test_loader = load_dataset(
        cfg.dataset.path,
        number_trajectories=cfg.dataset.number_trajectories,
        points_trajectory=cfg.dataset.points_trajectory,
        batchsize=cfg.dataset.batch_size,
        flatten_transitions=False,
    )

    # Extract ALL test trajectories
    trajectories = get_test_trajectories(
        test_loader,
        steps=cfg.model.steps,
    )

    all_metrics = []
    all_predictions = []

    diagnostics_dir = os.path.join(
        cfg.training.output_dir,
        "diagnostics",
    )
    os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"Evaluating {len(trajectories)} test trajectories")

    for i, (trajectory_index, true_states) in enumerate(trajectories):
        predicted_states = predict_trajectory(
            model,
            true_states,
            device,
            steps=cfg.model.steps,
        )

        true_states = true_states.cpu().numpy()

        # Only plot the first 10 trajectories
        make_plots = i < 10

        save_dir = os.path.join(
            diagnostics_dir,
            f"trajectory_{i + 1:03d}_dataset_{trajectory_index}",
        )

        metrics = run_diagnostics(
            model,
            true_states=true_states,
            predicted_states=predicted_states,
            dt=cfg.model.dt,
            device=device,
            save_dir=save_dir,
            model_type=cfg.model.type,
            steps=cfg.model.steps,
            make_plots=make_plots,
        )

        all_metrics.append(metrics)

        all_predictions.append(
            {
                "trajectory_index": trajectory_index,
                "true_states": true_states,
                "predicted_states": predicted_states,
            }
        )

        if i < 10:
            print(f"\nTrajectory {i + 1} (dataset index {trajectory_index}):")
            print(metrics)

    if not all_metrics:
        raise RuntimeError("No test trajectories found.")

    # Calculate aggregate metrics over ALL test trajectories
    print("\n========== FINAL TEST RESULTS (ALL TRAJECTORIES) ==========")

    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [
            metrics[metric_name]
            for metrics in all_metrics
            if np.isscalar(metrics[metric_name]) and np.isfinite(metrics[metric_name])
        ]

        if values:
            print(f"{metric_name}: {np.mean(values):.6e} +/- {np.std(values):.6e}")

    return all_predictions, all_metrics


@hydra.main(
    version_base=None,
    config_path="models/conf",
    config_name="config",
)
def main(cfg: DictConfig):

    if cfg.mode == "train":
        run_training(cfg)

    elif cfg.mode == "eval":
        _, _ = run_evaluation(cfg)

    elif cfg.mode == "train_eval":
        run_training(cfg)
        _, _ = run_evaluation(cfg)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
