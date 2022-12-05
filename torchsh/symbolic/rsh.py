"""Symbolic real spherical harmonic basis functions.

Methods in this module generate symbolic expressions for
real spherical harmonic functions of degree n and order m
in Cartesian form:
    Ynm: R^3 -> R
         x,y,z -> Ynm(x,y,z)

We use the Herglotzian definition to generate three polynomials
associated with x,y and z coordinates. See "Separated Cartesian form" [1]
for details.

The reason we don't use `sympy.Znm`, which takes spherical coordinates,
is that I did not manage to simplify/substitute the resulting expressions
enough so that they become simple functions of x,y,z respectively.

In the current implementation we use code adapted from
https://github.com/NVlabs/tiny-cuda-nn/blob/8e6e242f36dd197134c9b9275a8e5108a8e3af78/scripts/gen_sh.py
which uses a recurrent definition. It turned out to be more stable during tests.

References:
    - https://en.wikipedia.org/wiki/Spherical_harmonics#Separated_Cartesian_form
    - http://en.citizendium.org/wiki/Solid_harmonics#In_total
    - https://www.osti.gov/pages/servlets/purl/1172304
"""

try:
    import sympy as sym
except ImportError:
    raise Exception(
        "Failed to import sympy. You need `pip install sympy` for this part of the"
        " code."
    )


def _init_sin_cos_terms(n: int, x: sym.Symbol, y: sym.Symbol, z: sym.Symbol):
    S = [0]
    C = [1]
    for i in range(10):
        S.append(sym.simplify(x * S[i] + y * C[i]))
        C.append(sym.simplify(x * C[i] - y * S[i]))
    return S, C


def Ynm(n: int, m: int, x: sym.Symbol, y: sym.Symbol, z: sym.Symbol):
    """Return a symbolic expression for the real spherical
    harmonics having degree `l` and order `m`.

    Adapted from
    https://github.com/NVlabs/tiny-cuda-nn/blob/8e6e242f36dd197134c9b9275a8e5108a8e3af78/scripts/gen_sh.py
    """
    S, C = _init_sin_cos_terms(n, x, y, z)

    def K(n, m):
        return sym.sqrt(
            (2 * n + 1)
            * sym.factorial(n - abs(m))
            / (4 * sym.pi * sym.factorial(n + abs(m)))
        )

    def P(n, m):
        if n == 0 and m == 0:
            return 1
        if n == m:
            return (1 - 2 * m) * P(m - 1, m - 1)
        if n == m + 1:
            return (2 * m + 1) * z * P(m, m)
        return ((2 * n - 1) * z * P(n - 1, m) - (n + m - 1) * P(n - 2, m)) / (n - m)

    def Y(n, m):
        if m > 0:
            return sym.sqrt(2) * K(n, m) * C[m] * P(n, m)
        if m < 0:
            return sym.sqrt(2) * K(n, m) * S[-m] * P(n, -m)
        return K(n, m) * P(n, m)

    return Y(n, m)
