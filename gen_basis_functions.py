import sympy as sym
from sympy.functions.special.spherical_harmonics import Znm


theta, phi = sym.symbols("phi, theta", real=True)
x, y, z = sym.symbols("x:z", real=True)


def apply_subs(f: sym.Expr) -> sym.Expr:
    # f = f.simplify(inverse=True)
    # f = f.subs(theta, sym.acos(z))
    # f = f.subs(phi, sym.atan2(y, x))
    # f = f.simplify(inverse=True)
    f = f.simplify(inverse=True)
    f = f.subs(sym.sin(phi) * sym.sin(theta), y)
    f = f.simplify()
    f = f.subs(sym.sin(phi) * sym.cos(theta), x)
    f = f.simplify()
    f = f.subs(sym.cos(phi), z)
    f = f.simplify()
    return f


for l in range(0, 4):
    for m in range(-l, l + 1):
        f = sym.Znm(l, m, phi, theta).expand(func=True)
        print(f"Yn({l})m({m})={apply_subs(f)}")


# f = sym.Znm(1, 0, phi, theta).expand(func=True)
# f = f.simplify()
# f = f.subs(sym.sin(phi) * sym.sin(theta), y)
# f = f.subs(sym.sin(phi) * sym.cos(theta), x)
# f = f.subs(sym.cos(phi), z)
# print(f)
# print(sym.N(f))

# print(sym.simplify(sym.Znm(0, 0, theta, phi).expand(func=True)))
# print(sym.simplify())
