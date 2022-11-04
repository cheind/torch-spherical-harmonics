from typing import Callable


try:
    import sympy as sym
except ImportError:
    raise Exception(
        "Failed to import sympy\n. You need `pip install sympy` for this part of the"
        " code."
    )


class RenderLaTeX(object):
    pass


import sympy as sym
import torch

from .rsh import Ylm

x, y, z = sym.symbols("x:z", real=True)

_subs = {
    x**2: sym.symbols("x2"),
    y**2: sym.symbols("y2"),
    z**2: sym.symbols("z2"),
    x * y: sym.symbols("xy"),
    x * z: sym.symbols("xz"),
    y * z: sym.symbols("yz"),
}

code_tpl = r"""
import torch

def rsh_cart_{order}(xyz:torch.Tensor):
    '''Computes all real spherical harmonics up to a predefined order.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere
    
    Returns:
        rsh: (N,...,K) real spherical harmonics projections of input.
    '''

    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    x2=x**2
    y2=y**2
    z2=z**2
    xy=x*y
    xz=x*z
    yz=y*z

    return torch.stack(
        [{ynms}]
        ,-1
    )    
"""


def _substitute(f: sym.Expr) -> sym.Expr:
    """Substitutes pre-computed values of spherical harmonics"""
    return f.subs(_subs)


def generate_sh_fn_str(degree: int = 4, start: int = 0) -> str:
    """Returns the source code for `rsh_cart` method defined up to given order."""

    ynms = []
    for n in range(start, degree):
        for m in range(-n, n + 1):
            ylm = Ylm(n, m, x, y, z)
            ylmstr = sym.pycode(_substitute(sym.N(ylm)))
            if n == 0:
                ylmstr = f"xyz.new_tensor({ylmstr}).expand(xyz.shape[:-1])"
            ynms.append(ylmstr)
    return code_tpl.format(order=degree, ynms=",".join(ynms))


def compile_sh_fn(
    degree: int = 4, start: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:
    source = generate_sh_fn_str(degree=degree, start=start)
    ctx = {}
    exec(source, ctx)
    return ctx[f"rsh_cart_{degree}"]
