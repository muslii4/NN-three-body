import numpy as np
import torch


def _compute_loss(model, x, y, criterion, loss_function):
    """Compute either the standard or model-specific training loss."""
    if loss_function is None:
        return criterion(model(x), y)

    result = loss_function(x, y)
    if isinstance(result, tuple):
        return result[0]
    return result


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    loss_function=None,
):
    """Train a model for one epoch and return its sample-weighted loss.

    Args:
        model: PyTorch model to optimize.
        dataloader: Batches of input and target tensors.
        optimizer: Optimizer used to update model parameters.
        criterion: Loss function comparing predictions with targets.
        device: Device on which batches and the model are evaluated.
        loss_function: Optional model-specific loss function accepting
            ``(inputs, targets)``. It may return either a loss tensor or a
            ``(loss, components)`` tuple.

    Returns:
        The mean loss over all samples in ``dataloader``.
    """

    model.train()

    total_samples = 0
    total_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = _compute_loss(model, x, y, criterion, loss_function)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss = total_loss + loss.item() * batch_size
        total_samples = total_samples + batch_size

    return total_loss / total_samples


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    loss_function=None,
):
    """Evaluate a model for one epoch without updating its parameters.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Batches of input and target tensors.
        criterion: Loss function comparing predictions with targets.
        device: Device on which batches and the model are evaluated.
        loss_function: Optional model-specific loss function accepting
            ``(inputs, targets)``. It may return either a loss tensor or a
            ``(loss, components)`` tuple.

    Returns:
        The mean loss over all samples in ``dataloader``.
    """
    model.eval()
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            loss = _compute_loss(model, x, y, criterion, loss_function)
            batchsize = x.size(0)
            total_loss = total_loss + loss.item() * batchsize
            total_samples = total_samples + batchsize

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    """Collect model predictions and labels for every batch.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Batches of input and target tensors.
        device: Device on which batches and the model are evaluated.

    Returns:
        A pair of NumPy arrays containing predictions and corresponding labels.
    """
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_predictions = model(x)
            predictions.append(y_predictions.cpu().numpy())
            labels.append(y.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels
