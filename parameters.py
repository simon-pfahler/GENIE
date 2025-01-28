# Wilson operator parameters
config_file = f"test_gauge_fields/8c16/2200.config.npy"
mass = -0.5555

# GENIE network parameters
nr_layers = 8  # number of parallel transports

# training parameters
learning_rate = 1e-2
training_steps = 200
filter_iterations = 10

# output paths
weights_filename = "weights_GENIE.pt"
residuals_folder = "residuals"
