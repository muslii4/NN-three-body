"""Train a selected three-body model and report test metrics."""

import os
from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.baseline import PhysicsResNet
from models.MLP import MLP
from preprocessing import load_dataset
from training import evaluate, train_one_epoch, validate_one_epoch
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

    raise ValueError(f"Unknown model type: {cfg.model.type}")


def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_test_trajectory(test_loader, steps):
    """Extract one complete trajectory"""

    test_subset = test_loader.dataset
    trajectory_index = test_subset.indices[0]
    inputs, targets = test_subset.dataset[trajectory_index]

    true_states = torch.cat(
        (inputs[:-1], targets),
        dim=0,
    )

    return true_states[: steps + 1]


def predict_trajectory(model, true_states, device, steps):
    """Predict a complete trajectory from its initial state."""

    model.eval()
    initial_state = true_states[:1].to(device)

    with torch.no_grad():
        predicted = model.predict(initial_state, steps=steps)

    return predicted.squeeze(0).cpu().numpy()


def run_training(cfg: DictConfig):
    type_model = cfg.model.type

    early_stopper = EarlyStopper(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        max_overfit=cfg.training.early_stopping.max_overfit,
    )

    training_loader, validation_loader, test_loader = load_dataset(
        cfg.dataset.path,
        number_trajectories=cfg.dataset.number_trajectories,
        points_trajectory=cfg.dataset.points_trajectory,
        batchsize=cfg.dataset.batch_size,
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

    if cfg.model.type == "resnet":
        loss_function = model.loss_function

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
        )

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            loss_function=loss_function,
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

    predictions, labels = evaluate(
        model,
        test_loader,
        device,
    )

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)

    print(f"MSE={mse}, MAE={mae}")
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": cfg.model.type,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        os.path.join(cfg.training.output_dir, "model.pt"),
    )

    return (
        model,
        predictions,
        labels,
        test_loader,
        training_loader,
        validation_loader,
        device,
        type_model,
    )


def load_model(cfg: DictConfig, device: torch.device):
    """Load a trained model from a checkpoint."""

    checkpoint_path = os.path.join(cfg.training.output_dir, "model.pt")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_evaluation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(cfg, device)

    _, _, test_loader = load_dataset(
        cfg.dataset.path,
        number_trajectories=cfg.dataset.number_trajectories,
        points_trajectory=cfg.dataset.points_trajectory,
        batchsize=cfg.dataset.batch_size,
    )

    if cfg.model.type in ("resnet", "mlp"):
        true_states = get_test_trajectory(
            test_loader,
            steps=cfg.model.steps,
        )

        predicted_states = predict_trajectory(
            model,
            true_states,
            device,
            steps=cfg.model.steps,
        )

        true_states = true_states.cpu().numpy()

        metrics = run_diagnostics(
            model,
            true_states=true_states,
            predicted_states=predicted_states,
            dt=cfg.model.dt,
            device=device,
            model_type=cfg.model.type,
            steps=cfg.model.steps,
        )

        print(metrics)


@hydra.main(version_base=None, config_path="models/conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.mode == "train":
        run_training(cfg)
    elif cfg.mode == "eval":
        run_evaluation(cfg)
    elif cfg.mode == "train_eval":
        run_training(cfg)
        run_evaluation(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    # to run the model use
    # uv run main.py {model type}={mlp,resnet} {mode}={train,eval, train_eval} {steps}=model.steps=x + 1
    main()
