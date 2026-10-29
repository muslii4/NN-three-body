import numpy as np
import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model.train()

    total_samples = 0
    total_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss = total_loss + loss.item() * batch_size
        total_samples = total_samples + batch_size

    return total_loss / total_samples


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            batchsize = x.size(0)
            total_loss = total_loss + loss.item() * batchsize
            total_samples = total_samples + batchsize

    return total_loss / total_samples


def evaluate(model, dataloader, device):
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
