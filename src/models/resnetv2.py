"""Derivative-scaled physics-informed ResNet for the 2D three-body problem.

Unlike the first resnet, this model predicts a state
derivative and integrates it with the configured time step::

    x_next = x + dt * f_theta(x)

The output layer is zero-initialized, so training starts from the identity
transition. This keeps the network output O(1) instead of forcing it to learn
state increments of order ``dt``.
"""

import torch
import torch.nn.functional as F

from models.baseline import PhysicsResNet


class PhysicsResNetV2(PhysicsResNet):
    def __init__(
        self,
        input_size=12,
        hidden_size=256,
        num_layers=5,
        G=1.0,
        masses=(1.0, 1.0, 1.0),
        dt=1e-4,
        epsilon=1e-8,
        lambda_kinematics=0.1,
        lambda_dynamics=0.1,
        lambda_energy=0.01,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            G=G,
            masses=masses,
            dt=dt,
            epsilon=epsilon,
        )

        self.lambda_kinematics = lambda_kinematics
        self.lambda_dynamics = lambda_dynamics
        self.lambda_energy = lambda_energy

        # Start from the identity transition x_next = x.
        torch.nn.init.zeros_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """Predict the state after one time step as ``x + dt * f(x)``."""
        hidden = self.input_layer(x)
        hidden = self.residual_blocks(hidden)
        derivative = self.output_layer(hidden)

        return x + self.dt * derivative

    def data_loss(self, x, target, dt=None):
        """Mean squared error between predicted and target state derivatives."""
        if dt is None:
            dt = self.dt

        predicted_derivative = (self.forward(x) - x) / dt
        target_derivative = (target - x) / dt

        return F.mse_loss(predicted_derivative, target_derivative)

    def physics_loss(self, x, x_next, dt=None):
        """Compute physics residuals in derivative units."""
        if dt is None:
            dt = self.dt

        q0 = x[..., :6]
        v0 = x[..., 6:]
        q1 = x_next[..., :6]
        v1 = x_next[..., 6:]

        a0 = self.gravitational_acceleration(x)
        a1 = self.gravitational_acceleration(x_next)

        # Kinematic residual per unit time.
        q_residual = (q1 - q0) / dt - v0 - 0.5 * dt * a0
        loss_kinematics = torch.mean(q_residual**2)

        # Dynamic residual per unit time.
        v_residual = (v1 - v0) / dt - 0.5 * (a0 + a1)
        loss_dynamics = torch.mean(v_residual**2)

        # Energy residual is already dimensionless.
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

    def loss_function(self, x, target, dt=None):
        """Combine the supervised derivative loss with weighted physics terms."""
        if dt is None:
            dt = self.dt

        x_next = self.forward(x)

        predicted_derivative = (x_next - x) / dt
        target_derivative = (target - x) / dt
        loss_data = F.mse_loss(predicted_derivative, target_derivative)

        physics = self.physics_loss(x, x_next, dt)

        loss_total = (
            loss_data
            + self.lambda_kinematics * physics["kinematics"]
            + self.lambda_dynamics * physics["dynamics"]
            + self.lambda_energy * physics["energy_loss"]
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
