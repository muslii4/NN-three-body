import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset_generator import sample_ic
from generate_data import generate_data
from models.MLP import MLP
from preprocessing import load_dataset
from training import evaluate, train_one_epoch, validate_one_epoch

SEED = 42
EPOCHS_LEARNING = 100
EPOCHS_VALIDATION = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

rng = np.random.default_rng()
initial_x, initial_y, initial_vx, initial_vy = sample_ic(rng)

x, y, _, _ = generate_data(
    15,
    "verlet",
    initial_x,
    initial_y,
    initial_vx,
    initial_vy,
)
# print(x, y)
# plot_data(x, y)
# plot_data_animated(x, y)

"""
filepath = "dataset_1k_v1.h5"
with h5py.File(filepath, "r") as file:
    print(file.keys())
    print(file["positions"].shape)
    i = 0
    while True:
        positions = file["positions"][i]
        print(positions.shape)
        i += 1
"""

training_loader, val_loader, test_loader = load_dataset(
    "dataset_1k_v1.h5", batchsize=BATCH_SIZE
)

# To get all dataset:
# training_loader, val_loader, test_loader = load_dataset("dataset_1k_v1.h5", batchsize=BATCH_SIZE, number_trajectories=1000, points_trajectory=150001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS_LEARNING):
    train_loss = train_one_epoch(
        model=model,
        dataloader=training_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    print(f"Training Epoch {epoch + 1}, Loss: {train_loss:.6f}")

for epoch in range(EPOCHS_VALIDATION):
    validation_loss = validate_one_epoch(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Validation Epoch {epoch + 1}, Loss: {validation_loss:.6f}")

predictions, labels = evaluate(model=model, dataloader=test_loader, device=device)

predictions = predictions.reshape(-1)
labels = labels.reshape(-1)

mae = mean_absolute_error(labels, predictions)
mse = mean_squared_error(labels, predictions)

print(f"MSE={mse}, MAE={mae}")
