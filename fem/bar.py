"""1-D axial bar, 2-node linear elements.

Fixed at x=0, uniform distributed axial load q (N/m), EA constant.
Exact: u(x) = q/(EA) * (L x - x^2/2)
"""
import numpy as np


def solve_bar(n_el: int, L=1.0, EA=1.0, q=1.0):
    le = L / n_el
    K = np.zeros((n_el + 1, n_el + 1))
    F = np.zeros(n_el + 1)
    ke = EA / le * np.array([[1, -1], [-1, 1]])
    fe = q * le / 2 * np.array([1, 1])
    for e in range(n_el):
        K[e:e + 2, e:e + 2] += ke
        F[e:e + 2] += fe
    u = np.zeros(n_el + 1)
    u[1:] = np.linalg.solve(K[1:, 1:], F[1:])
    x = np.linspace(0, L, n_el + 1)
    return x, u


def exact(x, L=1.0, EA=1.0, q=1.0):
    return q / EA * (L * x - x ** 2 / 2)
