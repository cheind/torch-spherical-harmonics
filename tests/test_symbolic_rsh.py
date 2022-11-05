import pytest
import sympy as sym
import numpy as np
from numpy.testing import assert_allclose

from torchsh.symbolic.rsh import Ylm


@pytest.mark.parametrize("n,m", [(1, 0), (1, -1), (2, -1), (2, -2), (3, 2), (3, 1)])
def test_symbolic_rsh(n: int, m: int):
    # Numeric eval of Ynm
    x, y, z = sym.symbols("x,y,z")
    npf = sym.utilities.lambdify((x, y, z), Ylm(n, m, x, y, z), modules="numpy")
    # Comparison reference implementation using spherical coords

    theta, phi = sym.symbols("theta, phi")
    npf_ref = sym.utilities.lambdify(
        (theta, phi),
        sym.Znm(n, m, theta, phi).expand(func=True),
        modules=["numpy", "sympy"],
    )

    # Genereate coordinates

    theta_np = np.linspace(0, np.pi, 50)
    phi_np = np.linspace(0, 2 * np.pi, 50)
    theta_np, phi_np = np.meshgrid(theta_np, phi_np, indexing="ij")
    xyz = np.stack(
        [
            np.sin(theta_np) * np.cos(phi_np),
            np.sin(theta_np) * np.sin(phi_np),
            np.cos(theta_np),
        ],
        -1,
    )

    sh = npf(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    sh_ref = np.real(npf_ref(theta_np, phi_np))

    # Seems like sympy has an additional phase?
    # https://passthrough.fw-notify.net/download/284744/https://cs.dartmouth.edu/wjarosz/publications/dissertation/appendixB.pdf
    assert_allclose(abs(sh), abs(sh_ref), atol=1e-5)
