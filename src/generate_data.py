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

Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
InitialConditions = tuple[float, float, float]
IntegrationMethod = Literal["euler", "verlet"]


def _compute_a(
    x: Array1D,
    y: Array1D,
    M: Array1D,
    ax: Array2D,
    ay: Array2D,
    G: float,
) -> tuple[Array1D, Array1D]:
    """Compute the acceleration of each body at a given time."""
    ax_tot = ay_tot = np.zeros_like(M)
    # Loop over the three bodies
    for j in range(3):
        # Distance between the current body and the others (also with itself)
        dx = x[j] - x
        dy = y[j] - y

        # Acceleration according to Newton's law.
        # The division by zero for a body interacting with itself produces NaN;
        # it is handled below, so the warning is silenced.
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[:, j] = (-dx * M * G) / (np.sqrt(dx**2 + dy**2)) ** 3
            ay[:, j] = (-dy * M * G) / (np.sqrt(dx**2 + dy**2)) ** 3

        # Change NaN into zeros. The NaN comes from the interaction of a body
        # with itself, since (dx, dy) = 0.
        ax[np.isnan(ax)] = 0
        ay[np.isnan(ay)] = 0

        ax_tot = np.sum(ax, axis=0)
        ay_tot = np.sum(ay, axis=0)

    return ax_tot, ay_tot


def generate_data(
    length: float,
    method: IntegrationMethod = "verlet",
    initial_x: InitialConditions = (0.0, 0.0, 0.0),
    initial_y: InitialConditions = (1.0, -1.0, 0.0),
    initial_vx: InitialConditions = (0.3471, 0.3471, -0.6942),
    initial_vy: InitialConditions = (0.5327, 0.5327, -1.0654),
) -> tuple[Array2D, Array2D]:
    # Time step [s]
    step = 0.001

    # Body mass
    m1 = m2 = m3 = 1
    M = np.array([m1, m2, m3])

    # Newton constant
    G = 1

    t = np.arange(0, length + step, step)

    # Preparing the vector to store the solution.
    # There are 3 columns since we want to simulate three bodies.
    # One extra row is needed because the loop also stores x(i+1).
    x = np.zeros((len(t) + 1, 3))
    y = np.zeros((len(t) + 1, 3))

    vx = np.zeros((len(t) + 1, 3))
    vy = np.zeros((len(t) + 1, 3))

    ax = np.zeros((3, 3))
    ay = np.zeros((3, 3))

    # Assign the initial condition to the first element of the solution vectors
    x[0, :] = initial_x
    y[0, :] = initial_y
    vx[0, :] = initial_vx
    vy[0, :] = initial_vy

    if method not in ("euler", "verlet"):
        raise ValueError(f"Unknown integration method: {method}")

    for i in range(len(t)):
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

    return vx, vy


def plot_data_animated(
    x: Array2D,
    y: Array2D,
) -> animation.FuncAnimation:
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
        for line, body in zip(lines, bodies):
            line.set_data([], [])
            body.set_data([], [])
        return lines + bodies

    def update(frame: int) -> list[Line2D]:
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

    plt.show()
    return ani


def plot_data(
    x: Array2D,
    y: Array2D,
) -> None:
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.plot(x[:, 0], y[:, 0], "r")
    plt.plot(x[:, 1], y[:, 1], "g")
    plt.plot(x[:, 2], y[:, 2], "b")
    plt.legend(["Body 1", "Body 2", "Body 3"])
    plt.show()
