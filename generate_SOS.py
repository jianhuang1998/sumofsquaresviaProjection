"""
This file provides tools for generating sum-of-squares (SOS) polynomials and
associated auxiliary data for projecting onto the affine subspace corresponding to the SOS structure.

Functions:

* `general_sos`: Generates an SOS polynomial `f` in `n` variables of degree `2*d` with a specified number of terms. Coefficients are drawn from a uniform distribution by default, but can also be specified manually.

* `project_to_linear_space`: Project onto the affine subspace V(f),
  corresponding to the system Ax = b.

* `make_proj_vector_space_SOS`: Project onto the linear subspace spanned by the monomials of f,
  corresponding to the homogeneous system Ax = 0.
"""

import numpy as np
import random
import math
from numpy import *

def project_to_linear_space(Q, g, indices_matrix, nb_parts,full = None):
    """
    Project a given symmetric matrix Q onto the affine subspace V(f) 
    defined by the coefficient constraints of a sum-of-squares (SOS) polynomial.

    This function computes the orthogonal projection of the input matrix `Q` 
    onto the affine subspace corresponding to the SOS polynomial, as defined by 
    the linear equality constraints ⟨Q, B_α⟩ = f_α. 
    It serves as a core component in verifying SOS polynomial conditions via projection-based methods.

    Parameters
    ----------
    Q : numpy.ndarray
        The symmetric matrix to be projected.
    g : numpy.ndarray
        The vector of coefficients corresponding to the SOS polynomial.
    indices_matrix : numpy.ndarray
        The index mapping that encodes the linear relationships among polynomial coefficients.
    nb_parts : int
        The number of partitions used for averaging in the projection process.
    full : bool, optional, default=False
        If True, returns both the symmetrized projected matrix and the residual term. 
        If False, only the projected matrix is returned.

    Returns
    -------
    numpy.ndarray
        The projected (symmetrized) matrix Q.
    numpy.ndarray, optional
        The residual vector, returned only if `full=True`.

    Notes
    -----
    The affine subspace V(f) is defined as:
        V(f) = { Q ∈ S^{s(n,d)} | ⟨Q, B_α⟩ = f_α, ∀ α ∈ N_{2d}^n }.
    """
    BQ = np.zeros_like(g)
    for i, row in enumerate(indices_matrix):# Never twice the same index on a row
        BQ[row] += Q[i]
    projrem = (g - BQ) / nb_parts
    Qinterm = (Q + projrem[indices_matrix]) / 2
    if full:
        return Qinterm + Qinterm.T, g-BQ # To ensure that the numerical errors do not accumulate away from symmetry
    else:
        return Qinterm + Qinterm.T

def generate_polynomial_basis(d, n):  # d: degree n: dimension
    basis = []

    def generate_recursive(current_comb, current_sum, index):
        if index == n:
            if current_sum <= d:
                current_comb[-1] = d - current_sum
                basis.append(current_comb[:])
            return

        for i in range(d - current_sum + 1):
            current_comb[index] = i
            # pdb.set_trace()
            generate_recursive(current_comb, current_sum + i, index + 1)

    current_comb = [0] * (n + 1)
    generate_recursive(current_comb, 0, 0)

    basis.sort(key=lambda x: (sum(x), [-i for i in x]))
    return array(basis)


def nb_part_intern(degs, remainder, dic):
    if remainder == 0:
        return 1
    if len(degs) == 1:
        return 1

    try:
        return dic[(tuple(degs), remainder)]
    except KeyError:
        remdegs = degs[1:].sum()
        minnew = maximum(remainder - remdegs, 0)
        maxnew = minimum(remainder, degs[0])
        res = 0
        for new in range(minnew, maxnew + 1):
            res += nb_part_intern(degs[1:], remainder - new, dic)
        dic[(tuple(degs), remainder)] = res
        return res


def nb_part(degrees, remainder, dic):
    if remainder == 0:
        return 1
    if len(degrees) == 1:
        return 1
    degs = degrees.compress(degrees)
    degs.sort()
    degs = degs[::-1]
    try:
        return dic[(tuple(degs), remainder)]
    except KeyError:
        remdegs = degs[1:].sum()
        minnew = maximum(remainder - remdegs, 0)
        maxnew = minimum(remainder, degs[0])
        res = 0
        for new in range(minnew, maxnew + 1):
            res += nb_part_intern(degs[1:], remainder - new, dic)
        dic[(tuple(degs), remainder)] = res
        return res


def dictionary_basis(basis):

    return {tuple(base): i for i, base in enumerate(basis)}


# DEFINES THE AUXILIARY ELEMENTS.
def auxiliary(n, d, dic_nb={}):
    basis1 = generate_polynomial_basis(d, n)
    basis2 = generate_polynomial_basis(2 * d, n)
    d1 = dictionary_basis(basis1)
    d2 = dictionary_basis(basis2)

    indices_matrix = np.zeros((len(basis1), len(basis1)), int)
    for i in range(len(basis1)):
        for j in range(i, len(basis1)):
            coord = d2[tuple(basis1[i] + basis1[j])]
            indices_matrix[i, j] = coord
            indices_matrix[j, i] = coord

    nb_parts = [nb_part(degs, d, dic_nb) for degs in basis2]

    return basis1, basis2, d1, d2, indices_matrix, nb_parts


def generate_random_coeff(n, d):
    # Between -1 and 1
    dim_coeff = math.comb(n + d, d)
    return 2 * np.random.rand(dim_coeff) - 1


def g_from_Q(Q, indices_matrix):
    g = np.zeros(indices_matrix[-1, -1] + 1)
    for i, row in enumerate(indices_matrix):  # Never twice the same index on a row
        g[row] += Q[i]

    return g


def sos_from_polynoms_to_be_squared(list_coeffs_basis1, indices_matrix, weights=1, full=False):
    """
    list_coeffs_basis1 is a list where each element is the array of coefficients in the
    basis basis1, in the same order as generated by generate_polynomial_basis.
    """
    array_coeffs_basis1 = array(list_coeffs_basis1) * sqrt(weights).reshape(-1, 1)
    Q = einsum('ij, ik -> jk', array_coeffs_basis1, array_coeffs_basis1, optimize=True)

    BQ = g_from_Q(Q, indices_matrix)

    if full:
        return BQ, Q
    else:
        return BQ


def general_sos(n, d, qty, indices_matrix, full=False):
    """
    Generate random sum-of-squares (SOS) polynomials and their associated coefficient matrix.

    This function constructs SOS polynomials in n variables and degree 2d by
    randomly generating quadratic forms. Each polynomial is represented by
    a coefficient vector and, optionally, its corresponding Gram matrix.

    Parameters
    ----------
    n : int
        The number of variables in the polynomial.
    d : int
        The half-degree of the polynomial (the SOS polynomial has total degree 2d).
    qty : int
        The number of component polynomials whose squared sum forms the final SOS polynomial.
    indices_matrix : numpy.ndarray
        The index mapping that encodes the linear relationships among polynomial coefficients.
    full : bool, optional, default=False
        If True, returns the Gram matrices along with the generated coefficients.
        If False, only the final coefficient vector of the SOS polynomial is returned.

    Returns
    -------
    gfinal : numpy.ndarray
        The vector of coefficients of the resulting SOS polynomial.
    array_coeffs_basis1 : numpy.ndarray, optional
        The array of coefficients of each polynomial component (only if `full=True`).
    Q : numpy.ndarray, optional
        The corresponding Gram matrices of the SOS polynomial (only if `full=True`).

    Notes
    -----
    The generated SOS polynomial has the form:
        f(x) = ∑_{i=1}^{qty} (p_i(x))²,
    where each p_i(x) is a randomly generated polynomial of degree ≤ d.

    Examples
    --------
    >>> n, d, qty = 3, 2, 2
    >>> gfinal = general_sos(n, d, qty, indices_matrix)
    >>> gfinal, coeffs, Q = general_sos(n, d, qty, indices_matrix, full=True)
    """

    dim1 = len(indices_matrix)
    dim2 = indices_matrix[-1, -1] + 1

    array_coeffs_basis1 = 2 * np.random.rand(qty, dim1) - 1

    if full:
        gfinal, Q = sos_from_polynoms_to_be_squared(array_coeffs_basis1, indices_matrix, full=True)
        return gfinal, array_coeffs_basis1, Q

    else:
        gfinal = sos_from_polynoms_to_be_squared(array_coeffs_basis1, indices_matrix)
        return gfinal


def general_sos_monomial(n, d, qty, num_mono, indices_matrix,
                         full=False):  # qty: The number of polynomials that sum of square is SOS-polynomial.

    dim1 = len(indices_matrix)
    dim2 = indices_matrix[-1, -1] + 1

    array_coeffs_basis1 = 2 * np.random.rand(qty, dim1) - 1
    for i in range(qty):
        indices = random.sample(range(dim1), dim1 - num_mono)
        array_coeffs_basis1[i, indices] = 0

    if full:
        gfinal, Q = sos_from_polynoms_to_be_squared(array_coeffs_basis1, indices_matrix, full=True)
        return gfinal, array_coeffs_basis1, Q

    else:
        gfinal = sos_from_polynoms_to_be_squared(array_coeffs_basis1, indices_matrix)
        return gfinal


def make_proj_vector_space_SOS(indices_matrix, nb_parts, full=False):
    dim1 = len(nb_parts)

    def proj_vector_space_SOS(Q):

        BQ = np.zeros((dim1,))
        for i, row in enumerate(indices_matrix):  # Never twice the same index on a row
            BQ[row] += Q[i]
        projrem = (0 - BQ) / nb_parts
        Qinterm = (Q + projrem[indices_matrix]) / 2  # To avoid numerical errors away from symmetry
        return Qinterm + Qinterm.T

    def project_to_orthogonal_space(Q):
        BQ = np.zeros((dim1,))
        for i, row in enumerate(indices_matrix):  # Never twice the same index on a row
            BQ[row] += Q[i]
        entry_per_cat = BQ / nb_parts
        return entry_per_cat[indices_matrix]

    if full:
        return proj_vector_space_SOS, project_to_orthogonal_space
    else:
        return proj_vector_space_SOS
