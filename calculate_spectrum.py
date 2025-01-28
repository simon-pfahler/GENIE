import numpy as np
import qcd_ml
import torch
from scipy.sparse.linalg import LinearOperator, eigs

from GENIE_network import GENIE_network
from parameters import *

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


# calculate spectrum
def mw_np(x, shift):
    inp = torch.tensor(
        np.reshape(x, (*lattice_sizes, 4, 3)), dtype=torch.cdouble
    )
    with torch.no_grad():
        res = model.forward(w(inp)) + shift * inp
    return np.reshape(res.detach().numpy(), (np.prod(lattice_sizes) * 4 * 3))


shift = -10
mw_LinOp = LinearOperator(
    shape=(np.prod(lattice_sizes) * 4 * 3, np.prod(lattice_sizes) * 4 * 3),
    matvec=lambda x: mw_np(x, shift),
)
eigenvalues, eigenvectors = eigs(
    mw_LinOp, k=100, which="LM", return_eigenvectors=True, tol=1e-2
)
print("Lowest 100 eigenvalues of the preconditioned system (Re Im Tolerance):")
for i, eigenvalue in enumerate(eigenvalues):
    eigenvalue -= shift
    eigenvector = eigenvectors[:, i]
    applied_eigenvector = mw_np(eigenvector, 0)
    scaled_eigenvector = eigenvalue * eigenvector
    err = np.linalg.norm(applied_eigenvector - scaled_eigenvector)
    print(eigenvalue.real, eigenvalue.imag, err)
