"""Symbolic real spherical harmonic basis functions.

Methods in this module generate symbolic expressions for
real spherical harmonics in Cartesian form
    Ynm: R^3 -> R
         x,y,z -> Ynm(x,y,z)

We use the Herglotzian definition to generate three polynomials
associated with x,y and z coordinates. See "Separated Cartesian form" [1]
for details.


The reason we don't use `sympy.Znm`, which takes spherical coordinates,
is that I did not manage to simplify/substitute the resulting expressions
enough so that they become simple functions of x,y,z respectively.

References:
    - https://en.wikipedia.org/wiki/Spherical_harmonics#Separated_Cartesian_form
    - http://en.citizendium.org/wiki/Solid_harmonics#In_total
"""


import sympy as sym
import math


theta, phi = sym.symbols("phi, theta", real=True)
x, y, z = sym.symbols("x:z", real=True)


def A(m: int, x: sym.Symbol, y: sym.Symbol) -> sym.Expr:
    """A polynom associated with x,y"""
    p = sym.symbols("p", integer=True)
    f = sym.binomial(m, p) * x**p * y ** (m - p) * sym.cos((m - p) * sym.pi / 2)
    return sym.Sum(f, (p, 0, m))  # (sum is inclusive in sympy)


def B(m: int, x: sym.Symbol, y: sym.Symbol) -> sym.Expr:
    """B polynom associated with x,y"""
    p = sym.symbols("p", integer=True)
    f = sym.binomial(m, p) * x**p * y ** (m - p) * sym.sin((m - p) * sym.pi / 2)
    return sym.Sum(f, (p, 0, m))  # (sum is inclusive in sympy)


def P(n: int, m: int, z: sym.Symbol) -> sym.Expr:
    """Polynom associated with z"""
    k = sym.symbols("k", integer=True)
    scale = sym.sqrt(sym.factorial(n - m) / sym.factorial(n + m))
    phase = (-1) ** k
    gamma = phase * 2 ** (-n) * sym.binomial(n, k) * sym.binomial(2 * n - 2 * k, n)
    gamma = gamma * sym.factorial((n - 2 * k)) / sym.factorial(n - 2 * k - m)
    f = gamma * z ** (n - 2 * k - m)

    upper = int(math.floor((n - m) / 2))
    return scale * sym.Sum(f, (k, 0, upper))


def Ylm(n: int, m: int, x: sym.Symbol, y: sym.Symbol, z: sym.Symbol) -> sym.Expr:
    """Return a symbolic expression for the real spherical
    harmonics at degree `l` and index `m`."""
    assert abs(m) <= n
    if m < 0:
        f = sym.sqrt((2 * n + 1) / (2 * sym.pi)) * P(n, abs(m), z) * B(abs(m), x, y)
    elif m == 0:
        f = sym.sqrt((2 * n + 1) / (4 * sym.pi)) * P(n, 0, z)
    else:
        f = sym.sqrt((2 * n + 1) / (2 * sym.pi)) * P(n, m, z) * A(m, x, y)
    return f.doit().simplify()
