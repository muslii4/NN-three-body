import numpy as np

from dataset_generator import sample_ic
from generate_data import generate_data, plot_data

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
print(x, y)
plot_data(x, y)
# plot_data_animated(x, y)
