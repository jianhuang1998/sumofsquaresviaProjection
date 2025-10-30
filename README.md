# SOSviaProjection
# Code for the article "Verification of Sum-of-Squares Polynomials Using Projection Methods"

Projection-based methods for verifying sum-of-squares (SOS) polynomials.

**Dependencies for projection methods:** `numpy (1.26.4), scipy (1.13.1)`

**Extra dependencies for simulations in article:** `tables (3.9.1), sympy (1.13.1), SumOfSquares (1.3.1), MOSEK (10.1.13)`

The experiments are organized as follows:
* `generate_SOS.py` — generates SOS polynomials and constructs the operator for projection onto the affine subspace.
* `projection.py` — implements all projection-based methods proposed for verifying SOS polynomials.
* `Experiments.py` — runs comprehensive experiments on polynomial dimension and rank.
* `data_for_paper` - coefficients of the SOS polynomials and the experimental results reported in the paper.
* `All_the _experiments_inpaper.ipynb` - Reproduction of the article's experiments.

## Demo

Below is a simple example showing how to generate an SOS polynomial and verify it using the projection-based methods.

```python
# Import required functions
from generate_SOS import general_sos, make_proj_vector_space_SOS, project_to_linear_space, auxiliary
from projection import HIPswitch

# Ensure a SOS polynomial in 2 variables of degree 4 with 3 terms
basis1, basis2, d1, d2, indices_matrix, nb_parts = auxiliary(n=2, d=2)
dim = len(indices_matrix)
g, _, Q = general_sos(n=2, d=2, qty=3)

# Construct the projection operator onto the affine subspace V(f)
proj_vector_space_V = make_proj_vector_space_SOS(indices_matrix, nb_parts)
proj_0_on_V = project_to_linear_space(zeros((dim, dim)), g, indices_matrix, nb_parts)

# Verify the SOS polynomial using the HIPswitch projection method
data, nlevs = HIPswitch(proj_0_on_V, proj_vector_space_V, proj_0_on_V, maxiter=1000, tol=1e-8)
