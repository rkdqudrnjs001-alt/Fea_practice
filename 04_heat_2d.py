"""Step 4 - 2-D steady heat conduction with linear triangles + convergence."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fem.heat2d import exact, solve_heat

hs, errs = [], []
for n in [4, 8, 16, 32]:
    pts, tri, T = solve_heat(n)
    hs.append(1 / n)
    errs.append(np.sqrt(np.mean((T - exact(pts)) ** 2)))
rate = np.polyfit(np.log(hs), np.log(errs), 1)[0]
print(f"RMS nodal error: {['%.2e' % e for e in errs]}  -> convergence rate O(h^{rate:.2f})")

pts, tri, T = solve_heat(24)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
tr = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
c = ax[0].tricontourf(tr, T, 20, cmap="inferno")
ax[0].triplot(tr, lw=0.2, color="w", alpha=0.4)
fig.colorbar(c, ax=ax[0])
ax[0].set_aspect("equal")
ax[0].set_title("temperature (FEM, 24x24x2 triangles)")
ax[1].loglog(hs, errs, "o-")
ax[1].set_xlabel("h")
ax[1].set_ylabel("RMS error")
ax[1].set_title(f"convergence, slope = {rate:.2f}")
fig.tight_layout()
fig.savefig("figures/04_heat2d.png", dpi=130)
