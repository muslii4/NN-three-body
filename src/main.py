"""Train a selected three-body model and report test metrics."""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.baseline import PhysicsResNet
from models.MLP import MLP
from preprocessing import load_dataset
from training import evaluate, train_one_epoch, validate_one_epoch
from utilts import run_diagnostics


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
    """Extract one complete trajectory from the eager test dataset."""

    test_subset = test_loader.dataset
    trajectory_index = test_subset.indices[0]
    inputs, targets = test_subset.dataset[trajectory_index]

    true_states = torch.cat(
        (inputs[:1], targets),
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

    set_seed(cfg.seed)
    type_model = cfg.model.type
    training_loader, validation_loader, test_loader = load_dataset(
        cfg.dataset.path,
        batchsize=cfg.dataset.batch_size,
        points_trajectory=cfg.model.steps + 1,
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

    loss_function = None

    if cfg.model.type == "resnet":
        loss_function = model.loss_function

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device,
            loss_function=loss_function,
        )

        print(f"Training Epoch {epoch + 1}, Loss: {train_loss:.6f}")

    for epoch in range(cfg.training.validation_epochs):
        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            loss_function=loss_function,
        )

        print(f"Validation Epoch {epoch + 1}, Loss: {validation_loss:.6f}")

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
        points_trajectory=cfg.model.steps + 1,
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
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    # to run the model use
    # uv run main.py {model type}={mlp,resnet} {mode}={train,eval} {steps}=model.steps=x + 1
    main()
