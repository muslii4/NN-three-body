import torch
import torch.nn.functional as F
from torch import nn


class PhysicsResNet(nn.Module):
    """
    https://arxiv.org/pdf/2503.04585 inspired physics-informed ResNet for the 2D three-body problem.

    State:
    [x1, y1, x2, y2, x3, y3,
     vx1, vy1, vx2, vy2, vx3, vy3]

    Learns the complete state transition:
        x_next = x + N(x)

    Physics losses:
        1. Kinematic consistency
        2. Newtonian consistency
        3. Energy conservation
    """

    def __init__(
        self,
        input_size=12,
        hidden_size=256,
        num_layers=5,
        G=1.0,
        masses=(1.0, 1.0, 1.0),
        dt=1e-2,
        epsilon=1e-8,
    ):
        """Initialize the residual network and physical-system parameters.

        Args:
            input_size: Number of state features. The three-body state uses
                12 features.
            hidden_size: Width of each hidden layer.
            num_layers: Number of fully connected hidden layers.
            G: Gravitational constant used by the physics terms.
            masses: Masses of the three bodies.
            dt: Default time step for physics constraints.
            epsilon: Minimum distance used to avoid division by zero.
        """
        super().__init__()

        self.G = G
        self.masses = masses
        self.dt = dt
        self.eps = epsilon

        # Qin-style fully connected network.
        # Tanh hidden layers and linear output.
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])

        layers.append(nn.Linear(hidden_size, input_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Predict the complete state after one fixed time step.

        Args:
            x: State tensor with shape ``(batch, 12)`` or
                ``(batch, time, 12)``.

        Returns:
            Predicted state with the same shape as ``x``.
        """

        increment = self.network(x)

        return x + increment

    def gravitational_acceleration(self, x):
        """Compute Newtonian acceleration for each body.

        Args:
            x: State tensor whose first six features are positions.

        Returns:
            Tensor containing x and y acceleration for each body.
        """
        m1, m2, m3 = self.masses
        r12_vec, r13_vec, r23_vec = self.vector_between_points(x)

        r12 = torch.linalg.vector_norm(r12_vec, dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        r13 = torch.linalg.vector_norm(r13_vec, dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        r23 = torch.linalg.vector_norm(r23_vec, dim=-1, keepdim=True).clamp_min(
            self.eps
        )

        a1 = self.G * (m2 * r12_vec / r12**3 + m3 * r13_vec / r13**3)

        a2 = self.G * (-m1 * r12_vec / r12**3 + m3 * r23_vec / r23**3)

        a3 = self.G * (-m1 * r13_vec / r13**3 - m2 * r23_vec / r23**3)

        return torch.cat([a1, a2, a3], dim=-1)

    def vector_between_points(self, x):
        """Return pairwise displacement vectors between the bodies.

        Args:
            x: State tensor whose first six features are positions.

        Returns:
            Concatenated displacement vectors for pairs (1, 2), (1, 3), and
            (2, 3).
        """
        P = x[..., :6]

        p1 = P[..., 0:2]
        p2 = P[..., 2:4]
        p3 = P[..., 4:6]

        r12_vec = p2 - p1
        r13_vec = p3 - p1
        r23_vec = p3 - p2

        return r12_vec, r13_vec, r23_vec

    def kinetic_energy(self, x):
        """Compute kinetic energy for each state in a batch.

        Args:
            x: State tensor whose final six features are velocities.

        Returns:
            One kinetic-energy value for each leading batch element.
        """
        V = x[..., 6:]

        v1 = V[..., 0:2]
        v2 = V[..., 2:4]
        v3 = V[..., 4:6]

        m1, m2, m3 = self.masses

        return (
            0.5 * m1 * torch.sum(v1**2, dim=-1)
            + 0.5 * m2 * torch.sum(v2**2, dim=-1)
            + 0.5 * m3 * torch.sum(v3**2, dim=-1)
        )

    def potential_energy(self, x):
        """Compute gravitational potential energy for each state in a batch.

        Args:
            x: State tensor whose first six features are positions.

        Returns:
            One potential-energy value for each leading batch element.
        """
        m1, m2, m3 = self.masses
        r12_vec, r13_vec, r23_vec = self.vector_between_points(x)

        r12 = torch.linalg.vector_norm(r12_vec, dim=-1).clamp_min(self.eps)
        r13 = torch.linalg.vector_norm(r13_vec, dim=-1).clamp_min(self.eps)
        r23 = torch.linalg.vector_norm(r23_vec, dim=-1).clamp_min(self.eps)

        return -self.G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)

    def total_energy(self, x):
        """Compute total mechanical energy for each state in a batch.

        Args:
            x: State tensor containing positions and velocities.

        Returns:
            One total-energy value for each leading batch element.
        """
        return self.kinetic_energy(x) + self.potential_energy(x)

    def energy_consistency_loss(self, x, x_next):
        """Measure energy conservation between two consecutive states.

        Args:
            x: Current state tensor.
            x_next: Next state tensor.

        Returns:
            A dictionary containing the squared loss, absolute and relative
            residuals, and kinetic and potential energy changes.
        """
        T0 = self.kinetic_energy(x)
        V0 = self.potential_energy(x)

        T1 = self.kinetic_energy(x_next)
        V1 = self.potential_energy(x_next)

        delta_T = T1 - T0
        delta_V = V1 - V0

        # delta_T + delta_V = 0
        residual = delta_T + delta_V
        epsilon_abs = torch.abs(residual)

        epsilon_rel = epsilon_abs / (torch.abs(delta_T) + torch.abs(delta_V) + self.eps)

        # Squared residual for training
        loss_energy = torch.mean(residual**2)

        return {
            "energy_loss": loss_energy,
            "energy_abs": torch.mean(epsilon_abs),
            "energy_rel": torch.mean(epsilon_rel),
            "delta_T": torch.mean(delta_T),
            "delta_V": torch.mean(delta_V),
        }

    def physics_loss(self, x, x_next, dt=None):
        """Compute kinematic, dynamic, and energy physics residuals.

        Args:
            x: Current reference state.
            x_next: Predicted next state.
            dt: Optional override for the integration time step.

        Returns:
            Dictionary containing each physics loss and diagnostic metric.
        """

        if dt is None:
            dt = self.dt

        q0 = x[..., :6]
        v0 = x[..., 6:]
        q1 = x_next[..., :6]
        v1 = x_next[..., 6:]

        # Gravitational acceleration at both endpoints
        a0 = self.gravitational_acceleration(x)
        a1 = self.gravitational_acceleration(x_next)

        # Kinematic constraint:
        q_residual = q1 - q0 - 0.5 * dt * (v0 + v1)
        loss_kinematics = torch.mean(q_residual**2)

        # Dynamic constraint:
        v_residual = v1 - v0 - 0.5 * dt * (a0 + a1)
        loss_dynamics = torch.mean(v_residual**2)

        # Energy conservation
        energy = self.energy_consistency_loss(x, x_next)

        return {
            "kinematics": loss_kinematics,
            "dynamics": loss_dynamics,
            "energy_loss": energy["energy_loss"],
            "energy_abs": energy["energy_abs"],
            "energy_rel": energy["energy_rel"],
            "delta_T": energy["delta_T"],
            "delta_V": energy["delta_V"],
        }

    def loss_function(
        self,
        x,
        target,
        dt=None,
    ):
        """Combine supervised and physics-informed losses.

        Args:
            x: Reference current states.
            target: Reference next states.
            lambda_kin: Weight for the kinematic residual.
            lambda_dyn: Weight for the dynamic residual.
            lambda_energy: Weight for the energy residual.
            dt: Optional override for the integration time step.

        Returns:
            A tuple containing the total loss and a dictionary of components.
        """

        if dt is None:
            dt = self.dt

        # Predict next state
        x_next = self.forward(x)

        # Supervised data loss
        loss_data = F.mse_loss(x_next, target)

        # Physics losses evaluated on prediction
        physics = self.physics_loss(x, x_next, dt)

        loss_total = (
            loss_data
            + physics["kinematics"]
            + physics["dynamics"]
            + physics["energy_loss"]
        )

        components = {
            "total": loss_total,
            "data": loss_data,
            "kinematics": physics["kinematics"],
            "dynamics": physics["dynamics"],
            "energy_loss": physics["energy_loss"],
            "energy_abs": physics["energy_abs"],
            "energy_rel": physics["energy_rel"],
            "delta_T": physics["delta_T"],
            "delta_V": physics["delta_V"],
        }
        return loss_total, components

    def predict(self, initial_state, steps=1):
        """Roll the learned transition forward for a number of steps.

        Args:
            initial_state: Initial state tensor.
            steps: Number of transitions to predict.

        Returns:
            Tensor containing the initial state and all predicted states.
        """
        trajectory = [initial_state]
        x = initial_state

        for _ in range(steps):
            x = self.forward(x)
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)

    def learned_vector_field(self, x, dt=None):
        """Approximate the vector field implied by the learned transition.

        Args:
            x: Current state tensor.
            dt: Optional override for the integration time step.

        Returns:
            State derivative estimated from the predicted state increment.
        """
        if dt is None:
            dt = self.dt

        x_next = self.forward(x)

        return (x_next - x) / dt

    def true_vector_field(self, x):
        """Compute the analytical Newtonian vector field.

        Args:
            x: State tensor containing positions followed by velocities.

        Returns:
            Concatenated velocity and gravitational-acceleration tensors.
        """
        v = x[..., 6:]
        a = self.gravitational_acceleration(x)

        return torch.cat([v, a], dim=-1)
