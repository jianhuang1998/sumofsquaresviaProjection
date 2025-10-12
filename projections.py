"""
Available projection methods for verifying SOS polynomials:

* 'HIP_switch'           : Hyperplane Intersection Projection (HIP) with switching mode between pure HIP and alternating projection (AP).
* 'oneHIP_AP'            : Combination of HIP and AP; performs a few HIP steps followed by AP steps.
* 'pureHIP'              : Pure HIP method without switching or AP.
* 'AP_rank_nlev'         : Alternating Projection with truncated rank (APTR).
* 'lazy_AP_rank_nlev'    : Extrapolated version of APTR (extraAPTR).
* 'AP_fixed_rank_beta'   : Alternating Projection with fixed rank (APFR).
* 'lazy_AP_fixed_rank_beta': Extrapolated version of APFR (extraAPFR).
* 'mosek'                : Semidefinite programming solver Mosek for SOS verification.
"""

import numpy as N#function necessary
import scipy as S
import scipy.linalg as SL
import scipy.stats as SS
import scipy.sparse as SP
import scipy.optimize as SO
import scipy.sparse.linalg as SSL
# import piqp
import random
import cvxpy as cp
from types import SimpleNamespace
import random
import time

import numpy as np#function necessary
import math
from numpy import *


def step2(XW, target):
    nb_active = XW.shape[0]
    subset = N.array([nb_active - 1])
    coeffs = [target[-1] / XW[-1, -1]] # Always positive
    for i in range(nb_active - 2, -1, -1):
        #         if (N.all(XW[i, subset] <= 0)):###new test condition
        test = (XW[i, subset].dot(coeffs) < target[i])
        # The condition to project on the intersection of the hyperplanes is that
        # all the coefficients are non-negative. This is equivalent to belonging
        # to the normal cone to the facet.
        if test:
            subset = N.r_[i, subset]
            coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])
            # Adding a new hyperplane might generate negative coefficients.
            # We remove the corresponding hyperplanes, except if it is the last
            # hyperplane, in which case we do not add the hyperplane.
            if not N.all(coeffs >= 0):
                subset = subset[1:]
                coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])
    assert N.all(array(coeffs) >= 0  )####break here

    return subset, coeffs


def step_new(XW, target):
    nb_active = XW.shape[0]
    subset = N.array([nb_active - 1])
    coeffs = [target[-1] / XW[-1, -1]]  # Always positive
    for i in range(nb_active - 2, -1, -1):
        #         if (N.all(XW[i, subset] <= 0)):###new test condition
        # test = (XW[i, subset].dot(coeffs) < target[i])
        # The condition to project on the intersection of the hyperplanes is that
        # all the coefficients are non-negative. This is equivalent to belonging
        # to the normal cone to the facet.
        # if test:
        subseti = N.r_[i, subset]
        coeffsi = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])
        if N.all(array(coeffs) >= 0):
            subset = subseti
            coeffs = coeffsi
            # Adding a new hyperplane might generate negative coefficients.
            # We remove the corresponding hyperplanes, except if it is the last
            # hyperplane, in which case we do not add the hyperplane.
            # if coeffs[-1] < 0:
            #     subset = subset[1:]
            #     coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])
        elif not N.all(coeffs >= 0):
            subset = subset[N.where(coeffs >= 0)]
            coeffs = SL.inv(XW[N.ix_(subset, subset)]).dot(target[subset])

    # assert N.all(array(coeffs) >= 0)####break here

    return subset, coeffs


def HIP_switch(p, proj_vector_space_V, proj_0_on_V, max_mem_w = 30, min_cos =.99, maxiter=300, tol=1e-8)  :  # 7_30
    nlevs = []
    ps = []
    dim = len(p)
    dims = (dim, dim)
    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0 ,0))
    # lost_w = N.zeros(dims)
    w_act = N.zeros([0, dim, dim])
    target = N.array([])
    coeffs = N.array([])
    w_norm_ancien = N.zeros((dim, dim))
    p =  proj_0_on_V + proj_vector_space_V(p)
    sel = 'AP'

    for j in range(maxiter):
        print('Iteration' ,j)
        eigvals, eigvecs = SL.eigh(p)
        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break
            # rank = (eigvals > -enlarge).sum() ###########
        if sel == 'AP':
            print('AP mode')
            rank = (eigvals > 0).sum()#######

            APk = (eigvals[-rank:] * eigvecs[:,-rank:]) @ eigvecs[:,-rank:].conj().T
            # APk = proj_eps_widening_SPD(eps, enlarge, eigvals, eigvecs)
            # APk = find_best_on_ray(APk, proj_vector_space_V, proj_0_on_V)

            best_APk = proj_0_on_V + proj_vector_space_V(APk)
            w_new = best_APk - p
            norm_w = SL.norm(w_new)
            cos = N.vdot(w_new / norm_w, w_norm_ancien).real
            print("cos:" ,cos)
            change = (cos > min_cos)
            w_norm_ancien = w_new / norm_w
            if change:
                sel = 'HIP'
                active = N.array([j])
                nb_actives = 1
                XW = N.array([[norm_w**2]])
                w_act = N.array([w_new])
                target = N.array([SL.norm(p - APk)**2 / norm_w**2])
                coeffs = N.array([0.])
                p += coeffs[0] * w_new
            else:
                p = best_APk
                # active = N.array([])
                # nb_actives = 0
                # XW = N.zeros((0 ,0))
                # w_act = N.zeros([0, dim, dim])
                # target = N.array([])
                # coeffs = N.array([])

        elif sel == 'HIP':
            print(f'HIP mode. Active hyperplanes: {1 + nb_actives}')
            rank = (eigvals > 0).sum()#######

            APk = (eigvals[-rank:] * eigvecs[:,-rank:]) @ eigvecs[:,-rank:].conj().T
            # APk = proj_eps_widening_SPD(eps, enlarge, eigvals, eigvecs)
            # APk = find_best_on_ray(APk, proj_vector_space_V, proj_0_on_V)

            sq_norm_x_i = SL.norm(p - APk )**2
            w_i = proj_vector_space_V(APk) + proj_0_on_V - p
            xiwi = SL.norm(w_i )**2

            XW = N.column_stack([XW, N.zeros(nb_actives)])
            XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
            new_xw = N.einsum('ij, kij -> k', w_i.conj(), w_act).real # Notice that the scalar product are all real
            # since the matrices are self-adjoint.
            XW[-1, :-1] = new_xw
            XW[:-1, -1] = new_xw
            XW[-1, -1]  = xiwi
            target = N.r_[target, sq_norm_x_i]
            active = N.concatenate((active, [j]))
            w_act = N.concatenate([w_act, [w_i]])
            subset, coeffs = step2(XW, target)#can be repalced by step_new

            XW = XW[N.ix_(subset, subset)]
            active = active[subset]
            nb_actives = len(active)
            w_act = w_act[subset]
            # lost_w = w_act[0] / SL.norm(w_act[0])
            target = N.zeros((nb_actives,))
            p = p + N.einsum('k, kij -> ij', coeffs, w_act)

            if (subset[0] != 0) or nb_actives > max_mem_w:
                sel = 'AP'
                w_norm_ancien = N.zeros((dim, dim))
    return ps, nlevs


def oneHIP_AP(p, proj_vector_space_V, proj_0_on_V, oneHIP_steps=15, AP_steps=5, maxiter=300, tol=1e-8):
    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)

    cycle = oneHIP_steps + AP_steps
    for j in range(maxiter):
        print('Iteration' ,j)
        eigvals, eigvecs = SL.eigh(p)

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break
        if j % cycle < AP_steps:
            indices_positifs = (eigvals > 0)
            proj = (eigvecs[:, indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:, indices_positifs].T.conj()
            p = proj_vector_space_V(proj) + proj_0_on_V

        else:
            # oneHIP
            indices_positifs = (eigvals > 0)

            proj = (eigvecs[:, indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:, indices_positifs].T.conj()

            x = proj - p
            w = proj_vector_space_V(proj) + proj_0_on_V - p
            p += (SL.norm(x) ** 2 / SL.norm(w) ** 2) * w
            p = proj_vector_space_V(p) + proj_0_on_V  # Kills numerical errors
    ps.append(p)

    return ps, nlevs

def pureHIP(p, proj_vector_space_V, proj_0_on_V, maxiter=300, tol=1e-8):
    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)
    dim = len(p)
    dims = (dim, dim)
    active = N.array([])
    nb_actives = 0
    XW = N.zeros((0, 0))
    # lost_w = N.zeros(dims)
    w_act = N.zeros([0, dim, dim])
    target = N.array([])
    coeffs = N.array([])

    for j in range(maxiter):
        print(j)
        eigvals, eigvecs = SL.eigh(p)

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break
        rank = (eigvals > 0).sum()  #######

        APk = (eigvals[-rank:] * eigvecs[:, -rank:]) @ eigvecs[:, -rank:].conj().T
        # APk = proj_eps_widening_SPD(eps, enlarge, eigvals, eigvecs)
        # APk = find_best_on_ray(APk, proj_vector_space_V, proj_0_on_V)

        sq_norm_x_i = SL.norm(p - APk) ** 2
        w_i = proj_vector_space_V(APk) + proj_0_on_V - p
        xiwi = SL.norm(w_i) ** 2

        XW = N.column_stack([XW, N.zeros(nb_actives)])
        XW = N.row_stack([XW, N.zeros(nb_actives + 1)])
        new_xw = N.einsum('ij, kij -> k', w_i.conj(), w_act).real  # Notice that the scalar product are all real
        # since the matrices are self-adjoint.
        XW[-1, :-1] = new_xw
        XW[:-1, -1] = new_xw
        XW[-1, -1] = xiwi
        target = N.r_[target, sq_norm_x_i]
        active = N.concatenate((active, [j]))
        w_act = N.concatenate([w_act, [w_i]])
        subset, coeffs = step2(XW, target)

        XW = XW[N.ix_(subset, subset)]
        active = active[subset]
        nb_actives = len(active)
        w_act = w_act[subset]
        # lost_w = w_act[0] / SL.norm(w_act[0])
        target = N.zeros((nb_actives,))
        p = p + N.einsum('k, kij -> ij', coeffs, w_act)
    return ps, nlevs

def AP_rank_nlev(p, proj_vector_space_V, proj_0_on_V, maxiter=300, tol=1e-8):#APTR
    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)

    for j in range(maxiter):
        print(j)
        eigvals, eigvecs = SL.eigh(p)

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break

        if neg_least_ev > eigvals[-1] / 1.15:  # If no jump in evs, just usual AP
            rank = (eigvals > 0).sum()
            print(0, rank)
        else:
            rank = (eigvals > 1.1 * neg_least_ev).sum()
            print(-1, rank)

        APk = (eigvals[-rank:] * eigvecs[:, -rank:]) @ eigvecs[:, -rank:].conj().T
        # best_APk = find_best_on_ray(APk, proj_vector_space_V, proj_0_on_V)

        p = proj_0_on_V + proj_vector_space_V(APk)
    ps.append(p)

    return ps, nlevs


def lazy_AP_rank_nlev(p, proj_vector_space_V, proj_0_on_V, maxiter=300, tol=1e-8, partial_step=.9):
    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)

    for j in range(maxiter):
        print(j)
        eigvals, eigvecs = SL.eigh(p)

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break

        if neg_least_ev > eigvals[-1] / 1.15:  # If no jump in evs, just usual AP
            rank = (eigvals > 0).sum()
            print(0, rank)
        else:
            rank = (eigvals > 1.1 * neg_least_ev).sum()
            print(-1, rank)

        APk = (eigvals[-rank:] * eigvecs[:, -rank:]) @ eigvecs[:, -rank:].conj().T
        best_APk = proj_0_on_V + proj_vector_space_V(APk)

        if j % 5 == 4:  # Extrapolation step
            w1 = ps[-1] - ps[-2]
            print('w1n: ' + str(SL.norm(w1)))
            w2 = best_APk - ps[-1]
            print("w2n: " + str(SL.norm(w2)))
            alpha = np.vdot(w1, w2) / (w1 ** 2).sum()
            stepsize = partial_step / (1 - alpha)
            print('stepsize: ' + str(stepsize))
            p = ps[-1] + stepsize * w2
        else:
            p = best_APk

    ps.append(p)

    return ps, nlevs


def AP_fixed_rank_beta(p, proj_vector_space_V, proj_0_on_V, fixed_rank=1, maxiter=300, tol=1e-8):

    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)

    for j in range(maxiter):
        print(j)
        eigvals, eigvecs = SSL.eigsh(p, k=1, which='SA')

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break

        eigvals, eigvecs = SSL.eigsh(p, k=fixed_rank, which='LA')

        APk = (eigvals * eigvecs) @ eigvecs.conj().T
        best_APk = proj_vector_space_V(APk) + proj_0_on_V
        if j >= 2:
            print('cos:',
                  np.vdot(best_APk - ps[-1], ps[-1] - ps[-2]) / (SL.norm(best_APk - ps[-1]) * SL.norm(ps[-1] - ps[-2])))
        p = best_APk
    ps.append(p)
    return ps, nlevs


def lazy_AP_fixed_rank_beta(p, proj_vector_space_V, proj_0_on_V, fixed_rank=1, cos=0.99, maxiter=300, tol=1e-8):  # 7_30

    nlevs = []
    ps = []

    p = proj_0_on_V + proj_vector_space_V(p)

    for j in range(maxiter):
        print(j)
        eigvals, eigvecs = SSL.eigsh(p, k=1, which='SA')

        neg_least_ev = - eigvals[0]
        print(neg_least_ev)
        ps.append(p)
        nlevs.append(neg_least_ev)
        if neg_least_ev < tol:
            break

            # if neg_least_ev > eigvals[-1] / 1.15: # If no jump in evs, just usual AP
        # rank = (eigvals > enlarge).sum()
        # print(rank)
        # else:
        #     rank = (eigvals > 1.1 * neg_least_ev).sum()
        #     print(-1, rank)

        eigvals, eigvecs = SSL.eigsh(p, k=fixed_rank, which='LA')

        APk = (eigvals * eigvecs) @ eigvecs.conj().T

        best_APk = proj_0_on_V + proj_vector_space_V(APk)
        # best_APk = find_best_on_ray(APk, proj_vector_space_V, proj_0_on_V)
        # APk += (eigvals[:-rank] * enlarge) @ eigvecs[:,:-rank].conj().T
        if j % 5 == 4 and (np.vdot(best_APk - ps[-1], ps[-1] - ps[-2]) / (
                SL.norm(best_APk - ps[-1]) * SL.norm(ps[-1] - ps[-2])) > cos):  # Extrapolation step
            w1 = ps[-1] - ps[-2]
            print('w1n: ' + str(SL.norm(w1)))
            w2 = best_APk - ps[-1]
            print("w2n: " + str(SL.norm(w2)))
            print('cos:', np.vdot(w1, w2) / (SL.norm(w1) * SL.norm(w2)))
            alpha = np.vdot(w1, w2) / (w1 ** 2).sum()
            stepsize = .9 / (1 - alpha)
            print('stepsize: ' + str(stepsize))
            p = ps[-1] + stepsize * w2
            #### keep the rank
            # eigvals, eigvecs = SSL.eigsh(p,k = fixed_rank, which='LA')
            # APk = (eigvals * eigvecs) @ eigvecs.conj().T
            # best_APk = proj_0_on_V + proj_vector_space_V(APk)
        else:
            p = best_APk
        # if j>=2:
        #     print('cos:', np.vdot(best_APk - ps[-1],ps[-1]-ps[-2])/(SL.norm(best_APk - ps[-1])*SL.norm(ps[-1]-ps[-2])))
        # p = best_APk
    ps.append(p)
    return ps, nlevs

def generate_polynomial_basis(d, n):#d: degree n: dimension
    basis = []
    def generate_recursive(current_comb, current_sum, index):
        if index == n:
            if current_sum <= d:
                current_comb[-1] = d - current_sum
                basis.append(current_comb[:])
            return

        for i in range(d - current_sum + 1):
            current_comb[index] = i
            #pdb.set_trace()
            generate_recursive(current_comb, current_sum + i, index + 1)

    current_comb = [0] * (n + 1)
    generate_recursive(current_comb, 0, 0)

    basis.sort(key=lambda x: (sum(x), [-i for i in x]))
    return array(basis)

def from_coeff_to_polynomial(g,d,n):
    deg = generate_polynomial_basis(2*d,n)
    vars = sp.symbols(f'x:{n}')
    polynomial = 0
    for i in range(len(g)):
        term = g[i]
        for j in range(n):
            term *= vars[j] ** deg[i][1:][j]
        polynomial += term
    return sp.simplify(polynomial)
def mosek(g,d,n):
    p = from_coeff_to_polynomial(g,d,n)
    vars = sp.symbols(f'x:{n}')
    prob = SOSProblem()
    const = prob.add_sos_constraint(p, vars)
    try:
        prob.solve(solver='mosek',mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME': 1200}) # Raises SolutionFailure error due to infeasibility
        return 1
    except Exception as e:
        return 0
