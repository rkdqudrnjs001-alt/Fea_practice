"""2-D steady heat conduction (Laplace) on the unit square, linear triangles.

BC: T = sin(pi x) on the top edge (y = 1), T = 0 elsewhere.
Exact: T(x, y) = sin(pi x) sinh(pi y) / sinh(pi)
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def mesh(n):
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    tri = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 1, a + n + 2
            tri += [(a, b, d), (a, d, c)]
    return pts, np.array(tri)


def solve_heat(n, k=1.0):
    pts, tri = mesh(n)
    N = len(pts)
    K = lil_matrix((N, N))
    for t in tri:
        P = pts[t]
        B = np.array([[1, *P[0]], [1, *P[1]], [1, *P[2]]])
        area = 0.5 * abs(np.linalg.det(B))
        grads = np.linalg.inv(B)[1:, :]          # d(N_i)/dx, d(N_i)/dy
        ke = k * area * grads.T @ grads
        for i in range(3):
            for j in range(3):
                K[t[i], t[j]] += ke[i, j]
    K = K.tocsr()
    x, y = pts[:, 0], pts[:, 1]
    bnd = (x < 1e-12) | (x > 1 - 1e-12) | (y < 1e-12) | (y > 1 - 1e-12)
    Tb = np.where(y > 1 - 1e-12, np.sin(np.pi * x), 0.0)
    T = np.zeros(N)
    T[bnd] = Tb[bnd]
    free = ~bnd
    rhs = -K[free][:, bnd] @ T[bnd]
    T[free] = spsolve(K[free][:, free], rhs)
    return pts, tri, T


def exact(pts):
    x, y = pts[:, 0], pts[:, 1]
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
