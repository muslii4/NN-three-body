from generate_data import generate_data, plot_data_animated

x, y = generate_data(100, "verlet")
# plot_data(x, y)
plot_data_animated(x, y)
