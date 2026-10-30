"""Three bodies problem.

Translated from matlab core from:
andrea (2022). Three body problem
(https://www.mathworks.com/matlabcentral/fileexchange/96932-three-body-problem),
MATLAB Central File Exchange. Retrieved May 1, 2022.
"""

from typing import Literal

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numba import njit

Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
InitialConditions = tuple[float, float, float]
IntegrationMethod = Literal["euler", "verlet"]


@njit
def _compute_a(
    x: Array1D,
    y: Array1D,
    M: Array1D,
    ax: Array2D,
    ay: Array2D,
    G: float,
) -> tuple[Array1D, Array1D]:
    """Compute the acceleration of each body at a given time.

    Args:
        x: Current x coordinates of the bodies.
        y: Current y coordinates of the bodies.
        M: Body masses.
        ax: Reusable matrix for pairwise x accelerations.
        ay: Reusable matrix for pairwise y accelerations.
        G: Gravitational constant.

    Returns:
        Total x and y acceleration for each body.
    """
    for j in range(3):
        for i in range(3):
            if i == j:
                ax[i, j] = 0.0
                ay[i, j] = 0.0
            else:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r3 = np.sqrt(dx**2 + dy**2) ** 3
                ax[i, j] = (-dx * M[i] * G) / r3
                ay[i, j] = (-dy * M[i] * G) / r3

    return np.sum(ax, axis=0), np.sum(ay, axis=0)


@njit
def generate_data(
    length: float,
    method: IntegrationMethod = "verlet",
    initial_x: InitialConditions = (0.0, 0.0, 0.0),
    initial_y: InitialConditions = (1.0, -1.0, 0.0),
    initial_vx: InitialConditions = (0.3471, 0.3471, -0.6942),
    initial_vy: InitialConditions = (0.5327, 0.5327, -1.0654),
    step: float = 0.001,
) -> tuple[Array2D, Array2D, Array2D, Array2D]:
    """Integrate a three-body system with Euler or velocity Verlet.

    Args:
        length: Total simulated time.
        method: Integration method, either ``"euler"`` or ``"verlet"``.
        initial_x: Initial x coordinates for the three bodies.
        initial_y: Initial y coordinates for the three bodies.
        initial_vx: Initial x velocities for the three bodies.
        initial_vy: Initial y velocities for the three bodies.
        step: Integration time step.

    Returns:
        Four arrays of shape ``(n_steps + 1, 3)`` containing positions and
        velocities in the order ``x``, ``y``, ``vx``, ``vy``.

    Raises:
        ValueError: If ``method`` is not a supported integration method.
    """
    # Body mass
    m1 = m2 = m3 = 1.0
    M = np.array([m1, m2, m3])

    # Newton constant
    G = 1

    n_steps = round(length / step)

    # Preparing the vector to store the solution.
    # There are 3 columns since we want to simulate three bodies.
    # One extra row stores the initial state before integration.
    x = np.zeros((n_steps + 1, 3))
    y = np.zeros((n_steps + 1, 3))

    vx = np.zeros((n_steps + 1, 3))
    vy = np.zeros((n_steps + 1, 3))

    ax = np.zeros((3, 3))
    ay = np.zeros((3, 3))

    # Assign the initial condition to the first element of the solution vectors
    x[0, :] = initial_x
    y[0, :] = initial_y
    vx[0, :] = initial_vx
    vy[0, :] = initial_vy

    if method not in ("euler", "verlet"):
        raise ValueError(f"Unknown integration method: {method}")

    for i in range(n_steps):
        ax_tot, ay_tot = _compute_a(x[i, :], y[i, :], M, ax, ay, G)

        if method == "verlet":
            x[i + 1, :] = x[i, :] + step * vx[i, :] + 1 / 2 * (ax_tot * step**2)
            y[i + 1, :] = y[i, :] + step * vy[i, :] + 1 / 2 * (ay_tot * step**2)

            ax_tot_next, ay_tot_next = _compute_a(
                x[i + 1, :], y[i + 1, :], M, ax, ay, G
            )
            vx[i + 1, :] = vx[i, :] + 1 / 2 * (ax_tot + ax_tot_next) * step
            vy[i + 1, :] = vy[i, :] + 1 / 2 * (ay_tot + ay_tot_next) * step
        else:
            x[i + 1, :] = x[i, :] + step * vx[i, :]
            y[i + 1, :] = y[i, :] + step * vy[i, :]
            vx[i + 1, :] = vx[i, :] + step * ax_tot
            vy[i + 1, :] = vy[i, :] + step * ay_tot

    return x, y, vx, vy


def plot_data_animated(
    x: Array2D,
    y: Array2D,
) -> animation.FuncAnimation:
    """Create and save an animated plot of three-body trajectories.

    The animation is saved as ``three_body.gif`` in the current directory.

    Args:
        x: X-coordinate history with shape ``(time, 3)``.
        y: Y-coordinate history with shape ``(time, 3)``.

    Returns:
        The configured Matplotlib animation.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Three-Body Problem")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # Keep the viewing area fixed throughout the animation
    margin = 1
    ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)

    colors = ["r", "g", "b"]
    labels = ["Body 1", "Body 2", "Body 3"]

    # Lines show the paths travelled by each body
    lines = []
    for i in range(3):
        lines.append(ax.plot([], [], color=colors[i], label=labels[i])[0])

    # Dots show the current positions of the bodies
    bodies = []
    for i in range(3):
        bodies.append(ax.plot([], [], "o", color=colors[i], markersize=8)[0])

    ax.legend()

    def init() -> list[Line2D]:
        """Clear trajectory lines and body markers before animation starts."""
        for line, body in zip(lines, bodies):
            line.set_data([], [])
            body.set_data([], [])
        return lines + bodies

    def update(frame: int) -> list[Line2D]:
        """Update trajectory lines and body markers for one frame."""
        for i, (line, body) in enumerate(zip(lines, bodies)):
            line.set_data(x[: frame + 1, i], y[: frame + 1, i])
            body.set_data([x[frame, i]], [y[frame, i]])

        return lines + bodies

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(x), 30),
        init_func=init,
        interval=10,
        blit=True,
        repeat=True,
    )

    # plt.show()
    ani.save("three_body.gif", fps=30, writer="pillow")
    return ani


def plot_data(
    x: Array2D,
    y: Array2D,
) -> None:
    """Plot the three trajectories in a static Matplotlib figure.

    Args:
        x: X-coordinate history with shape ``(time, 3)``.
        y: Y-coordinate history with shape ``(time, 3)``.
    """
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.plot(x[:, 0], y[:, 0], "r")
    plt.plot(x[:, 1], y[:, 1], "g")
    plt.plot(x[:, 2], y[:, 2], "b")
    plt.legend(["Body 1", "Body 2", "Body 3"])
    plt.show()
