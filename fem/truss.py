"""2-D pin-jointed truss (bar elements, 2 dof per node)."""
import numpy as np


def warren_truss(n_bays=6, span=6.0, height=1.0):
    """Bottom chord nodes 0..n, top chord nodes above bay centres."""
    bx = np.linspace(0, span, n_bays + 1)
    tx = 0.5 * (bx[:-1] + bx[1:])
    nodes = np.vstack([np.column_stack([bx, np.zeros_like(bx)]),
                       np.column_stack([tx, np.full_like(tx, height)])])
    nb = n_bays + 1
    els = [(i, i + 1) for i in range(n_bays)]                       # bottom chord
    els += [(nb + i, nb + i + 1) for i in range(n_bays - 1)]        # top chord
    els += [(i, nb + i) for i in range(n_bays)]                     # diagonals
    els += [(i + 1, nb + i) for i in range(n_bays)]
    return nodes, np.array(els)


def solve_truss(nodes, els, EA, supports, loads):
    """supports: {node: (fix_x, fix_y)}, loads: {node: (Fx, Fy)}."""
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    geo = []
    for a, b in els:
        dx, dy = nodes[b] - nodes[a]
        L = np.hypot(dx, dy)
        c, s = dx / L, dy / L
        k = EA / L * np.outer([-c, -s, c, s], [-c, -s, c, s])
        idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        K[np.ix_(idx, idx)] += k
        geo.append((idx, c, s, L))
    for node, (fx, fy) in loads.items():
        F[2 * node] += fx
        F[2 * node + 1] += fy
    fixed = [2 * nd + i for nd, fl in supports.items() for i in (0, 1) if fl[i]]
    free = np.setdiff1d(np.arange(2 * n), fixed)
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    R = K @ u - F
    N = np.array([EA / L * np.dot([-c, -s, c, s], u[idx]) for idx, c, s, L in geo])
    return u.reshape(-1, 2), N, R.reshape(-1, 2)
