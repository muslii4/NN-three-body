"""Train a selected three-body model and report test metrics."""

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.baseline import PhysicsResNet
from models.MLP import MLP
from preprocessing import load_dataset
from training import evaluate, train_one_epoch, validate_one_epoch


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


@hydra.main(version_base=None, config_path="models/conf", config_name="config")
def run_training(cfg: DictConfig):

    set_seed(cfg.seed)

    training_loader, validation_loader, test_loader = load_dataset(
        cfg.dataset.path,
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


if __name__ == "__main__":
    run_training()
