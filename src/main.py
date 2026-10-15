from generate_data import generate_data, plot_data, plot_data_animated, verlet

x, y = generate_data(20, verlet)
plot_data(x, y)
plot_data_animated(x, y)
