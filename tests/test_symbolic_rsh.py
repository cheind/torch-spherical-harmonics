import pytest
import sympy as sym
import numpy as np
from numpy.testing import assert_allclose

from torchsh.symbolic.rsh import Ynm


@pytest.mark.parametrize("n,m", [(1, 0), (1, -1), (2, -1), (2, -2), (3, 2), (3, 1)])
def test_symbolic_rsh(n: int, m: int):
    # Numeric eval of Ynm
    x, y, z = sym.symbols("x,y,z")
    npf = sym.utilities.lambdify((x, y, z), Ynm(n, m, x, y, z), modules="numpy")
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


@pytest.mark.parametrize("n,m", [(1, 0), (1, -1), (2, -1), (3, 2), (3, 1)])
def test_symbolic_rsh_orthonormalized(n: int, m: int):
    # Numeric eval of Ynm
    x, y, z = sym.symbols("x,y,z")
    theta, phi = sym.symbols("theta, phi")
    ynm = Ynm(n, m, x, y, z)
    ynm = ynm.subs(
        {
            x: sym.sin(theta) * sym.cos(phi),
            y: sym.sin(theta) * sym.cos(phi),
            z: sym.cos(theta),
        }
    )

    i = sym.integrate(
        ynm * ynm * sym.sin(theta), (phi, 0, 2 * sym.pi), (theta, 0, sym.pi)
    )
    assert i.evalf() - 1 < 1e-5

    import random

    ynm_other = Ynm(n + 1, random.randint(-(n + 1), (n + 1)), x, y, z)
    i = sym.integrate(
        ynm * ynm_other * sym.sin(theta), (phi, 0, 2 * sym.pi), (theta, 0, sym.pi)
    )
    assert i.evalf() - 0 < 1e-5


# sym.integrate(f2*f2*sym.sin(theta),(phi,0,2*sym.pi), (theta,0,sym.pi))
