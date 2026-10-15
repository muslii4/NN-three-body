"""Three bodies problem.

Translated from matlab core from:
andrea (2022). Three body problem
(https://www.mathworks.com/matlabcentral/fileexchange/96932-three-body-problem),
MATLAB Central File Exchange. Retrieved May 1, 2022.
"""

import numpy as np
from matplotlib import pyplot as plt


def _compute_a(x, y, M, ax, ay, G):
    """Compute the acceleration of each body at a given time."""
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


def verlet(t, x, y, M, ax, ay, G, step, vx, vy) -> tuple[np.ndarray, np.ndarray]:
    for i in range(len(t)):
        # Compute the acceleration at the current time
        ax_tot, ay_tot = _compute_a(x[i, :], y[i, :], M, ax, ay, G)

        # Compute the next time position given the current acceleration
        x[i + 1, :] = x[i, :] + step * vx[i, :] + 1 / 2 * (ax_tot * step**2)
        y[i + 1, :] = y[i, :] + step * vy[i, :] + 1 / 2 * (ay_tot * step**2)

        # Compute the next time acceleration
        ax_tot_next, ay_tot_next = _compute_a(x[i + 1, :], y[i + 1, :], M, ax, ay, G)

        # Compute the next time velocity according to the Verlet method
        vx[i + 1, :] = vx[i, :] + 1 / 2 * (ax_tot + ax_tot_next) * step
        vy[i + 1, :] = vy[i, :] + 1 / 2 * (ay_tot + ay_tot_next) * step

    return vx, vy


def euler(t, x, y, M, ax, ay, G, step, vx, vy) -> tuple[np.ndarray, np.ndarray]:
    # Loop over each time step
    for i in range(len(t)):
        # Compute the current acceleration
        ax_tot, ay_tot = _compute_a(x[i, :], y[i, :], M, ax, ay, G)

        # Update the position
        x[i + 1, :] = x[i, :] + step * vx[i, :]
        y[i + 1, :] = y[i, :] + step * vy[i, :]

        # Update the velocity
        vx[i + 1, :] = vx[i, :] + step * ax_tot
        vy[i + 1, :] = vy[i, :] + step * ay_tot

    return vx, vy


def generate_data(length: float, method: callable) -> tuple[np.ndarray, np.ndarray]:
    # Time step [s]
    step = 0.001

    # Body mass
    m1 = m2 = m3 = 1
    M = np.array([m1, m2, m3])

    # Newton constant
    G = 1

    t = np.arange(0, length + step, step)

    # Initial conditions -- positions
    # Initial conditions were taken from:
    #   Numerical Search for Periodic Solutions in the Vicinity of the
    #   Figure-Eight Orbit: Slaloming around Singularities on the Shape Sphere
    #   Milovan Suvakov
    #   C. Moore, Phys. Rev. Lett. 70, 3675 (1993).
    #   C. Simo, Contemp Math 292 20928 (2002).

    x10 = 0  # -1
    y10 = 1  # 0

    x20 = -x10
    y20 = -y10

    x30 = 0
    y30 = 0

    # Initial conditions -- velocity
    vx10 = 0.3471
    vy10 = 0.5327

    vx20 = vx10
    vy20 = vy10

    vx30 = -2 * vx10
    vy30 = -2 * vy10

    # Simple Euler method

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
    x[0, 0] = x10
    x[0, 1] = x20
    x[0, 2] = x30

    y[0, 0] = y10
    y[0, 1] = y20
    y[0, 2] = y30

    vx[0, 0] = vx10
    vx[0, 1] = vx20
    vx[0, 2] = vx30

    vy[0, 0] = vy10
    vy[0, 1] = vy20
    vy[0, 2] = vy30

    return method(t, x, y, M, ax, ay, G, step, vx, vy)


def plot_data(x, y):
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.plot(x[:, 0], y[:, 0], "r")
    plt.plot(x[:, 1], y[:, 1], "g")
    plt.plot(x[:, 2], y[:, 2], "b")
    plt.legend(["Body 1", "Body 2", "Body 3"])
    plt.show()
