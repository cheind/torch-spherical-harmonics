import math
import torch
import sympy as sym
import numpy as np
from numpy.testing import assert_allclose

from torchsh.symbolic.codegen import compile_sh_fn


def test_compile_fn():
    rsh_fn = compile_sh_fn(degree=4)
    assert rsh_fn(torch.randn(10, 5, 3)).shape == (10, 5, 16)

    rsh_fn = compile_sh_fn(degree=4, start=1)
    assert rsh_fn(torch.randn(10, 5, 3)).shape == (10, 5, 15)

    # rsh_fn = compile_sh_fn(order=8)
    # assert rsh_fn(torch.randn(10, 5, 3)).shape == (10, 5, 64)


def test_compare_compiled_ref():
    # Numeric eval of Ynm
    theta, phi = sym.symbols("theta, phi")

    # Genereate coordinates

    theta_np = np.linspace(0, np.pi, 50)
    phi_np = np.linspace(0, 2 * np.pi, 50)
    theta_np, phi_np = np.meshgrid(theta_np, phi_np)
    theta_phi_np = np.stack((theta_np, phi_np), -1)
    xyz = np.stack(
        [
            np.sin(theta_np) * np.cos(phi_np),
            np.sin(theta_np) * np.sin(phi_np),
            np.cos(theta_np),
        ],
        -1,
    )

    rsh_fn = compile_sh_fn(degree=4)
    sh = rsh_fn(torch.tensor(xyz))
    idx = 1
    for n in range(1, 4):
        for m in range(-n, n + 1):
            npf_ref = sym.utilities.lambdify(
                (theta, phi),
                sym.Znm(n, m, theta, phi).expand(func=True),
                modules=["numpy", "sympy"],
            )
            sh_ref = torch.tensor(
                np.real(npf_ref(theta_phi_np[..., 0], theta_phi_np[..., 1]))
            )
            assert_allclose(
                abs(sh[..., idx]),
                abs(sh_ref),
                atol=1e-5,
            )

            idx += 1
