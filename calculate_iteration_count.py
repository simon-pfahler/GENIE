import os

import numpy as np
import qcd_ml
import torch
from scipy.sparse.linalg import LinearOperator, eigs

from GENIE_network import GENIE_network
from parameters import *

# define an inner product
innerproduct = lambda x, y: (x.conj() * y).sum()

# load the gauge field and define the Wilson operator
try:
    U = torch.load(config_file, weights_only=True)
except:
    raise RuntimeError("Loading of the gauge field failed")
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

# lattice sizes
lattice_sizes = U.shape[1:5]

# create the model and load the weights
model = GENIE_network(U, nr_layers)
model.load_state_dict(torch.load(weights_filename, weights_only=True))

# calculate iteration count
torch.manual_seed(43)
test_vectors = [
    torch.randn(*lattice_sizes, 4, 3, dtype=torch.cdouble) for _ in range(10)
]

test_vectors = [e / innerproduct(e, e) ** 0.5 for e in test_vectors]

if not os.path.exists(residuals_folder):
    os.makedirs(residuals_folder)

its = np.zeros(len(test_vectors))
with torch.no_grad():
    for i, test_vector in enumerate(test_vectors):
        _, ret_p = qcd_ml.util.solver.GMRES(
            w,
            test_vector,
            torch.zeros_like(test_vector),
            preconditioner=lambda x: model.forward(x),
            eps=1e-8,
            maxiter=10000,
        )
        its[i] = ret_p["k"]
        np.savetxt(
            os.path.join(residuals_folder, f"residuals_sample{i}.dat"),
            ret_p["history"],
        )
print(
    f"Model Iteration count: {np.mean(its)} +- "
    + f"{np.std(its, ddof=1)/np.sqrt(len(test_vectors))}"
)
