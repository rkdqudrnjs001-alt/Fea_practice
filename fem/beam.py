"""Euler-Bernoulli cantilever, 2-node Hermite elements (w, theta per node).

Loads: uniform q (N/m) and/or tip force P (N). Clamped at x = 0.
Exact tip deflection: qL^4/(8EI) + PL^3/(3EI); root moment: qL^2/2 + PL (sign: positive for load acting in +w).
"""
import numpy as np


def element_stiffness(EI, le):
    return EI / le ** 3 * np.array([
        [12, 6 * le, -12, 6 * le],
        [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
        [-12, -6 * le, 12, -6 * le],
        [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])


def solve_cantilever(n_el, L=1.0, EI=1.0, q=0.0, P=0.0):
    le = L / n_el
    ndof = 2 * (n_el + 1)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    fe = q * np.array([le / 2, le ** 2 / 12, le / 2, -le ** 2 / 12])   # consistent load
    for e in range(n_el):
        d = slice(2 * e, 2 * e + 4)
        K[d, d] += element_stiffness(EI, le)
        F[d] += fe
    F[-2] += P
    u = np.zeros(ndof)
    u[2:] = np.linalg.solve(K[2:, 2:], F[2:])
    # element end moments from the element stiffness relation
    M = []
    for e in range(n_el):
        d = slice(2 * e, 2 * e + 4)
        fint = element_stiffness(EI, le) @ u[d] - fe
        M.append(-fint[1])           # moment at the element's left node
    x = np.linspace(0, L, n_el + 1)
    return x, u[0::2], u[1::2], np.array(M)


def exact_deflection(x, L=1.0, EI=1.0, q=0.0, P=0.0):
    wq = q * x ** 2 / (24 * EI) * (6 * L ** 2 - 4 * L * x + x ** 2)
    wp = P * x ** 2 / (6 * EI) * (3 * L - x)
    return wq + wp


def exact_moment(x, L=1.0, q=0.0, P=0.0):
    return q * (L - x) ** 2 / 2 + P * (L - x)
