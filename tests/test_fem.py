import numpy as np

from fem.bar import exact as bar_exact, solve_bar
from fem.beam import exact_deflection, exact_moment, solve_cantilever
from fem.heat2d import exact as heat_exact, solve_heat
from fem.truss import solve_truss, warren_truss


def test_bar_nodal_exact():
    x, u = solve_bar(5)
    assert np.allclose(u, bar_exact(x))


def test_beam_matches_closed_form():
    x, w, _, M = solve_cantilever(6, L=2.0, EI=3.0, q=1.5, P=0.7)
    assert np.allclose(w, exact_deflection(x, 2.0, 3.0, 1.5, 0.7))
    assert np.allclose(M, exact_moment(x[:-1], 2.0, 1.5, 0.7))


def test_truss_equilibrium_and_symmetry():
    nodes, els = warren_truss(6)
    loads = {i: (0.0, -1.0) for i in range(1, 6)}
    u, N, R = solve_truss(nodes, els, 1e3, {0: (1, 1), 6: (0, 1)}, loads)
    assert np.isclose(R[:, 1].sum(), 5.0)
    assert np.isclose(R[0, 1], R[6, 1])            # symmetric reactions


def test_heat_second_order_convergence():
    e = []
    for n in (8, 16):
        pts, _, T = solve_heat(n)
        e.append(np.sqrt(np.mean((T - heat_exact(pts)) ** 2)))
    assert e[0] / e[1] > 3.0                       # ~4 for O(h^2)
