"""Step 3 - 2-D Warren truss bridge: global assembly, supports, member forces."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from fem.truss import solve_truss, warren_truss

nodes, els = warren_truss(n_bays=6, span=6.0, height=1.0)
nb = 7
loads = {i: (0.0, -10e3) for i in range(1, nb - 1)}            # 10 kN on inner bottom nodes
supports = {0: (1, 1), nb - 1: (0, 1)}                          # pin + roller
EA = 210e9 * 2e-3                                               # steel, 20 cm^2
u, N, R = solve_truss(nodes, els, EA, supports, loads)

print(f"sum of applied loads  = {sum(f[1] for f in loads.values())/1e3:.1f} kN")
print(f"sum of reactions      = {R[:, 1].sum()/1e3:.1f} kN  (equilibrium check)")
print(f"max |member force|    = {np.abs(N).max()/1e3:.1f} kN")
print(f"mid-span deflection   = {u[3, 1]*1e3:.2f} mm")

scale = 0.3 / np.abs(u).max()
norm = colors.TwoSlopeNorm(vmin=N.min(), vcenter=0, vmax=N.max())
fig, ax = plt.subplots(figsize=(8, 2.8))
for (a, b), n in zip(els, N):
    ax.plot(*nodes[[a, b]].T, color="0.8", lw=1)
    d = nodes + scale * u
    ax.plot(*d[[a, b]].T, color=cm.coolwarm(norm(n)), lw=3)
sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")
fig.colorbar(sm, ax=ax, label="axial force (N)  - compression / + tension")
ax.set_aspect("equal")
ax.set_title(f"deformed shape (x{scale:.0f}) coloured by member force")
fig.tight_layout()
fig.savefig("figures/03_truss.png", dpi=130)
