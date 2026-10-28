from torch import Tensor, nn


class MLP(nn.Module):
    """
    Class defining a general MLP model.
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128) -> None:
        """
        Function initializing the architecture of the model.

        Args:
        input_dim (int): dimenssion of the input, which we expect to be 12 (=3 (bodies) * 2 (coordinates) * 2 (parameters: velocity, position))
        hidden_dim (int): number of neurons in each hidden layer
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the network's forward pass.

        Args:
        x (Tensor): Input tensor that is passed.

        Returns:
        Tensor with the predicted output of the forward pass.
        """
        return self.network(x)
