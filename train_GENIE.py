import numpy as np
import qcd_ml
import torch
from scipy.sparse.linalg import LinearOperator, eigs

from GENIE_network import GENIE_network
from parameters import *

torch.manual_seed(42)

# use GPU if available
if torch.cuda.is_available():
    to_device = lambda x: x.cuda()
    torch.set_default_device("cuda")
    device = "cuda"
    print("CUDA is available and will be used.")
else:
    to_device = lambda x: x
    print("CUDA is not available!")

torch.set_num_threads(1)

# define an inner product
innerproduct = lambda x, y: (x.conj() * y).sum()

# load the gauge field and define the Wilson operator
try:
    U = to_device(torch.load(config_file, weights_only=True))
except:
    raise RuntimeError("Loading of the gauge field failed")
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

# lattice sizes
lattice_sizes = U.shape[1:5]

# create the model and initialize the weights
model = GENIE_network(U, nr_layers)
for li in model.dense_layers:
    li.weights.data = 0.001 * torch.randn_like(
        li.weights.data, dtype=torch.cdouble
    )

# define the filtering function
pre_filter = lambda x: qcd_ml.util.solver.GMRES(
    w,
    torch.zeros_like(x),
    x,
    maxiter=filter_iterations,
    inner_iter=filter_iterations,
    eps=1e-15,
    innerproduct=innerproduct,
)[0]

# training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
cost = np.zeros(training_steps)
print("Iteration - cost")
for t in range(training_steps):
    random_source = torch.randn(*lattice_sizes, 4, 3, dtype=torch.cdouble)
    filtered_source = pre_filter(random_source)

    filtered_source /= innerproduct(filtered_source, filtered_source) ** 0.5

    err = model.forward(w(filtered_source)) - filtered_source
    curr_cost = innerproduct(err, err).real

    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()
    print(f"{t} - {cost[t]}")

# save weights
torch.save(model.state_dict(), weights_filename)
