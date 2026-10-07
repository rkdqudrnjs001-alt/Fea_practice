"""Step 1 - 1-D axial bar: assembly, boundary conditions, convergence."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fem.bar import exact, solve_bar

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
xf = np.linspace(0, 1, 200)
ax[0].plot(xf, exact(xf), "k--", label="exact")
errs, ns = [], [2, 4, 8, 16, 32]
for n in ns:
    x, u = solve_bar(n)
    if n in (2, 4):
        ax[0].plot(x, u, "o-", ms=4, label=f"{n} elements")
    xm = 0.5 * (x[:-1] + x[1:])                      # mid-element error (interpolation)
    um = 0.5 * (u[:-1] + u[1:])
    errs.append(np.max(np.abs(um - exact(xm))))
ax[0].set_xlabel("x")
ax[0].set_ylabel("u(x)")
ax[0].legend(fontsize=8)
ax[1].loglog(1 / np.array(ns), errs, "o-")
rate = np.polyfit(np.log(1 / np.array(ns)), np.log(errs), 1)[0]
ax[1].set_title(f"max mid-element error, slope = {rate:.2f}")
ax[1].set_xlabel("element size h")
fig.tight_layout()
fig.savefig("figures/01_bar.png", dpi=130)
print(f"nodal values exact for linear elements; mid-element error converges at O(h^{rate:.1f})")
