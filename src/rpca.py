"""Solve the robust PCA problem via the Inexact ALM method."""

import numpy as np
from scipy import linalg as la


def shrink(M, tau):
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)


def svd_threshold(M, tau):
    U, s, VT = la.svd(M, full_matrices=False)
    return np.dot(U*shrink(s, tau), VT)


def ialm(M, lbd=None, mu=None, rho=1.6, tol1=1.0e-7, tol2=1.0e-5, max_iter=1000, iter_print=100, verbose=False):
    m, n = M.shape

    # ||M||_2 := largest singular value of M
    M2 = la.norm(M, ord=2)
    # ||M||_inf := max(abs(M))
    Minf = np.max(np.abs(M))
    # ||M||_F := Frobenius norm
    MF = la.norm(M, ord='fro')

    if lbd is None:
        lbd = 1.0 / np.sqrt(max(m, n))
        if verbose:
            print 'lbd:', lbd

    if mu is None:
        mu = 1.25 / M2
        if verbose:
            print 'mu:', mu

    Y = M / max(M2, Minf / lbd)
    L = np.zeros_like(M)
    S_old = np.zeros_like(M)

    for it in xrange(max_iter):
        S = shrink(M - L + Y/mu, lbd/mu)
        L = svd_threshold(M - S + Y/mu, 1.0/mu)
        Z = M - L - S

        e1 = la.norm(Z, ord='fro') / MF
        e2 = mu * la.norm(S - S_old, ord='fro') / MF
        cond1 = e1 < tol1
        cond2 = e2 < tol2
        if verbose and it % iter_print == 0:
            print 'Iteration {0}, mu = {1}, err1 = {2} -> {3}, err2 = {4} -> {5}'.format(it, mu, e1, tol1, e2, tol2)
        if cond1 and cond2:
            if verbose and it % iter_print != 0:
                print 'Iteration {0}, mu = {1}, err1 = {2} -> {3}, err2 = {4} -> {5}'.format(it, mu, e1, tol1, e2, tol2)
            break
        else:
            Y += mu * Z

        S_old = S
        if cond2:
            mu *= rho
    else:
        print 'Warn: Exit with max_iter = {0}, mu = {1}, err1 = {2} -> {3}, err2 = {4} -> {5}'.format(it, mu, e1, tol1, e2, tol2)

    return L, S