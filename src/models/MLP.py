import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """
    Class defining a general MLP model.
    """

    def __init__(
        self, input_dim: int = 12, hidden_dim: int = 128, dt: float = 1e-4
    ) -> None:
        """
        Function initializing the architecture of the model.

        Args:
        input_dim (int): dimenssion of the input, which we expect to be 12 (=3 (bodies) * 2 (coordinates) * 2 (parameters: velocity, position))
        hidden_dim (int): number of neurons in each hidden layer
        dt (float): time step used to integrate the predicted derivative
        """
        super().__init__()

        self.dt = dt

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the network's forward pass.

        Args:
        x (Tensor): Input tensor that is passed.

        Returns:
        Tensor with the predicted output of the forward pass.
        """
        return x + self.dt * self.network(x)

    def data_loss(self, x: Tensor, target: Tensor, dt: float | None = None) -> Tensor:
        """Compute the supervised derivative loss."""
        if dt is None:
            dt = self.dt

        predicted_derivative = self.network(x)
        target_derivative = (target - x) / dt

        return torch.nn.functional.mse_loss(predicted_derivative, target_derivative)

    def predict(self, initial_state: Tensor, steps: int = 1) -> Tensor:
        """Predict a trajectory by applying the n-step transition forward."""
        trajectory = [initial_state]
        state = initial_state

        for _ in range(steps):
            state = self.forward(state)
            trajectory.append(state)

        return torch.stack(trajectory, dim=1)
