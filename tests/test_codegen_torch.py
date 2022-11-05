import pytest
import torch
import sympy as sym
import numpy as np
from numpy.testing import assert_allclose

from torchsh.symbolic.codegen import compile_fn

from torchsh import rsh


def test_compile_fn():
    rsh_fn = compile_fn(degree=3)
    assert rsh_fn(torch.randn(10, 5, 3)).shape == (10, 5, 16)

    # rsh_fn = compile_sh_fn(order=8)
    # assert rsh_fn(torch.randn(10, 5, 3)).shape == (10, 5, 64)


@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_compare_compiled_ref(compile, degree):
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
    # If compile, we actually test the current state of codegen
    # Otherwise we test the pre-compiled functions
    if compile:
        rsh_fn = compile_fn(degree=degree)
    else:
        rsh_fn = getattr(rsh, f"rsh_cart_{degree}")
    sh = rsh_fn(torch.tensor(xyz))
    idx = 1  # We skip coefficient 0, since Znm does not return a array in this case
    for n in range(1, degree + 1):
        for m in range(-n, n + 1):
            print(n, m)
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
