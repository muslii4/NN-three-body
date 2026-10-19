from math import hypot

import numpy as np
from numba import njit

from generate_data import Array2D, InitialConditions, generate_data

PAIRS = [(0, 1), (0, 2), (1, 2)]
SEQUENCE_TIME = 15.0
MAX_DISTANCE_FROM_ORIGIN = 20.0
MAX_ENERGY_DIFF = 1e-6
MIN_PAIRWISE_DISTANCE = 0.1
MAX_PAIRWISE_DISTANCE = 20.0

MAX_INIT_POS = 2.0
MAX_INIT_VEL = 1.0

STEP = 1e-4


def sample_ic(
    rng: np.random.Generator,
    m: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pos_range: float = MAX_INIT_POS,
    vel_range: float = MAX_INIT_VEL,
    d_min: float = 0.2,
) -> tuple[InitialConditions, InitialConditions, InitialConditions, InitialConditions]:
    masses = np.asarray(m)
    while True:
        x1 = np.array([1.0, 0.0])
        x2 = rng.uniform(-pos_range, pos_range, size=2)
        x3 = -(masses[0] * x1 + masses[1] * x2) / masses[2]

        v1 = rng.uniform(-vel_range, vel_range, size=2)
        v2 = rng.uniform(-vel_range, vel_range, size=2)
        v3 = -(masses[0] * v1 + masses[1] * v2) / masses[2]

        positions = np.stack([x1, x2, x3])
        velocities = np.stack([v1, v2, v3])

        # reject close encounters at t=0
        dists = [
            np.linalg.norm(positions[i] - positions[j])
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ]
        if min(dists) < d_min:
            continue
        return (
            (
                float(positions[0, 0]),
                float(positions[1, 0]),
                float(positions[2, 0]),
            ),
            (
                float(positions[0, 1]),
                float(positions[1, 1]),
                float(positions[2, 1]),
            ),
            (
                float(velocities[0, 0]),
                float(velocities[1, 0]),
                float(velocities[2, 0]),
            ),
            (
                float(velocities[0, 1]),
                float(velocities[1, 1]),
                float(velocities[2, 1]),
            ),
        )


@njit
def total_energy(
    px: InitialConditions,
    py: InitialConditions,
    vx: InitialConditions,
    vy: InitialConditions,
):
    kinetic = 0
    kinetic += sum([v**2 for v in vx])
    kinetic += sum([v**2 for v in vy])
    kinetic /= 2

    potential = 0
    n = len(px)
    for i in range(n):
        for j in range(i + 1, n):
            potential -= 1 / hypot(px[i] - px[j], py[i] - py[j])

    return kinetic + potential


def pair_distances(px, py):
    return np.stack(
        [np.hypot(px[..., i] - px[..., j], py[..., i] - py[..., j]) for i, j in PAIRS],
        axis=-1,
    )


def generate_valid_sequence(rng) -> tuple[Array2D, Array2D, Array2D, Array2D]:
    while True:
        ix, iy, ivx, ivy = sample_ic(rng)
        initial_e = total_energy(ix, iy, ivx, ivy)
        if initial_e >= 0:
            continue
        x, y, vx, vy = generate_data(SEQUENCE_TIME, "verlet", ix, iy, ivx, ivy, STEP)

        end_e = total_energy(x[-1], y[-1], vx[-1], vy[-1])
        delta_e = abs(initial_e - end_e)
        if delta_e / abs(initial_e) > MAX_ENERGY_DIFF:
            print("Skipping: Energy not conserved")
            continue
        elif x.max() > MAX_DISTANCE_FROM_ORIGIN or y.max() > MAX_DISTANCE_FROM_ORIGIN:
            print("Skipping: bodies went too far")
            continue

        distances = pair_distances(x, y)
        if distances.min() < MIN_PAIRWISE_DISTANCE:
            print("Skipping: bodies went too close to each other")
            continue
        elif distances.max() > MAX_PAIRWISE_DISTANCE:
            print("Skipping: bodies went too far from each other")
            continue

        return x, y, vx, vy
