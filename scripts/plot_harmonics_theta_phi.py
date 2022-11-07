import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import count

# The following import configures Matplotlib for 3D plotting.
from torchsh.symbolic.codegen import compile_fn

# Degree to generate
DEGREE = 6


def plot_Y_theta_phi(ax, Y: torch.Tensor, n: int, m: int, index: int):
    sh = Y[..., index]

    img = ax.imshow(
        sh,
        extent=(0, 2 * np.pi, np.pi, 0),
        aspect=2,
        origin="upper",
        cmap="bwr",
    )
    img.set_clim(-0.5, 0.5)
    ax.set_xlabel(r"$\phi$", labelpad=-12, fontsize=8)
    ax.set_ylabel(r"$\theta$", labelpad=-12, fontsize=8)
    ax.set_xticks([0, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$2\pi$"], fontsize=6)
    ax.set_yticks([0, np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi$"], fontsize=6)
    if m >= 0:
        title_str = f"$Y_{{{n}{{,}}{m}}}$"
    else:
        title_str = f"$Y_{{{n}{{,}}{{-}}{abs(m)}}}$"
    ax.set_title(title_str, fontsize=10)
    return img


def main():

    theta = torch.linspace(0, np.pi, 100)
    phi = torch.linspace(0, 2 * np.pi, 100)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    xyz = torch.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        -1,
    )

    # Compile our torch harmonics function
    rsh_fn = compile_fn(degree=DEGREE)
    # Eval with cartesian coords
    Y = rsh_fn(xyz)

    # Setup the figure and grid
    fig = plt.figure(figsize=(21, 11), dpi=100)
    ncols = 2 * DEGREE + 1
    spec = gridspec.GridSpec(
        ncols=ncols + 1,
        nrows=DEGREE,
        figure=fig,
        wspace=0.3,
        hspace=0.4,
        width_ratios=[1.0] * ncols + [0.1],
    )

    # Plot
    idx = count()
    for n in range(DEGREE):
        for m in range(-n, n + 1):
            ax = fig.add_subplot(spec[n, m + DEGREE])
            img = plot_Y_theta_phi(ax, Y, n, m, next(idx))

    plt.colorbar(img, cax=plt.subplot(spec[:, -1]), orientation="vertical")
    plt.figtext(
        0.5,
        0.025,
        "https://github.com/cheind/torch-spherical-harmonics",
        ha="center",
        fontsize=8,
        fontfamily="monospace",
    )
    fig.savefig("etc/rsph_theta_phi.svg", facecolor="white")
    fig.savefig("etc/rsph_theta_phi.png", facecolor="white", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
