import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import count

# The following import configures Matplotlib for 3D plotting.
from torchsh.symbolic.codegen import compile_sh_fn

degree = 4

rsh_fn = compile_sh_fn(degree=degree)

theta = torch.linspace(0, np.pi, 100)
phi = torch.linspace(0, 2 * np.pi, 100)
theta, phi = torch.meshgrid(theta, phi, indexing="xy")
xyz = torch.stack(
    [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ],
    -1,
)

sh_coeffs = rsh_fn(xyz)


def plot_sh(ax, xyz, sh_coeffs, index: int):
    sh = sh_coeffs[..., index]

    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("jet"))
    # cmap.set_clim(-1.0, 1.0)
    ax.plot_surface(
        xyz[..., 0],
        xyz[..., 1],
        xyz[..., 2],
        facecolors=cmap.to_rgba(sh),
        linewidth=0,
        rstride=1,
        cstride=1,
    )
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])


# https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FReal-spherical-harmonics-up-to-degree-3-from-high-function-values-in-yellow-to-low_fig2_321331471&psig=AOvVaw0Vy6mrkQiu_tGvwc-vAAkK&ust=1667651168094000&source=images&cd=vfe&ved=0CA0QjRxqFwoTCLiv0u3ClPsCFQAAAAAdAAAAABAO


fig = plt.figure(figsize=(10, 10))
spec = gridspec.GridSpec(
    ncols=2 * degree + 1, nrows=degree, figure=fig, wspace=0, hspace=0
)

idx = count()
for n in range(degree):
    for m in range(-n, n + 1):
        ax = fig.add_subplot(spec[n, m + degree], projection="3d")
        plot_sh(ax, xyz, sh_coeffs, next(idx))
# plt.tight_layout()
# plt.savefig("sph_harm.png")
plt.show()
