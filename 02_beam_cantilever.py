"""Step 2 - Euler-Bernoulli cantilever with Hermite elements."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fem.beam import exact_deflection, exact_moment, solve_cantilever

L, EI, q, P = 1.0, 2.0, 3.0, 1.0
x, w, th, M = solve_cantilever(8, L, EI, q, P)
tip_ex = q * L ** 4 / (8 * EI) + P * L ** 3 / (3 * EI)
print(f"tip deflection  FEM={w[-1]:.6f}  exact={tip_ex:.6f}")
print(f"root moment     FEM={M[0]:.4f}  exact={exact_moment(0, L, q, P):.4f}")

# Element-end moments from the stiffness relation are exact at nodes
# (superconvergence). Inside an element the Hermite interpolation gives a
# linear curvature, so the mid-element moment EI*(theta2-theta1)/le converges.
ns, err = [1, 2, 4, 8, 16, 32], []
for n in ns:
    xx, _, thn, Mn = solve_cantilever(n, L, EI, q, P)
    assert np.allclose(Mn, exact_moment(xx[:-1], L, q, P))
    le = L / n
    m_mid = EI * np.abs(np.diff(thn)) / le
    err.append(np.max(np.abs(m_mid - exact_moment(xx[:-1] + le / 2, L, q, P))))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
xf = np.linspace(0, L, 200)
ax[0].plot(xf, exact_deflection(xf, L, EI, q, P), "k--", label="exact")
ax[0].plot(x, w, "o", label="FEM (8 el.)")
ax[0].set_title("deflection")
ax[0].legend()
ax[1].plot(xf, exact_moment(xf, L, q, P), "k--")
ax[1].step(x[:-1], M, where="post", label="FEM element-end moment")
ax[1].set_title("bending moment")
ax[1].legend(fontsize=8)
rate = np.polyfit(np.log(L / np.array(ns)), np.log(err), 1)[0]
ax[2].loglog(L / np.array(ns), err, "o-")
ax[2].set_title(f"mid-element moment error, slope = {rate:.2f}")
print(f"mid-element moment converges at O(h^{rate:.1f}); nodal values exact")
ax[2].set_xlabel("h")
fig.tight_layout()
fig.savefig("figures/02_beam.png", dpi=130)
