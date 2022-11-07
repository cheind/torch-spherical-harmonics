## **torch-spherical-harmonics**

Real spherical harmonics (RSH) in Cartesian form for PyTorch. The resulting source code is automatically created by converting optimized symbolic RSH expressions to PyTorch.

The following plot shows the first real spherical harmonics $Y_{nm}$ of degree $n < 6$ and order $-n \le m \le n$ as a function of polar coordinates $\theta \in [0,\pi]$ and $\phi \in [0,2\pi]$.
![](etc/rsph_theta_phi.png?raw=true)

## Properties

The RSH functions provided in this package are ortho-normalized over the surface of the unit sphere. That is

$$
\int\limits_0^\pi\int\limits_0^{2\pi}Y_{nm}(\theta,\phi)Y_{n'm'}(\theta,\phi)\sin\theta\,d\phi\,d\theta = \delta_{ll'}\delta_{mm'}.
$$

## Usage

To compute all $Y_{nm}$ values up to degree 3 use:

```python

import torch
import torchsh

xyz = ... # tensor (N,...,3) of points on the unit-sphere
sh = torchsh.rsh_cart_3(xyz) # tensor (N,...,16) of Ynm
```

This relies on pre-generated RSH functions, which `torchsh` contains up to degree 8. They follow the the naming convention `rsh_cart_{degree}`.

If you do not want to include a new library, you may just as well just include [`torchsh/rsh.py`](./torchsh/rsh.py) in your project, which requires only `torch` to be installed.

## Code Generation

We use `sympy` to generate RSH expressions in Cartesian form reyling on the [Herglotzian](https://en.wikipedia.org/wiki/Spherical_harmonics#Separated_Cartesian_form) definition. These expressions are simplified and transformed into Python/PyTorch functions using a code template and the string engine `mako`. In the tests we use `sympy.Znm` to verify our numerical results.

We initially intended to use `sympy.Znm` directly for code generation, by substituting polar coordinate definitions with respective Cartesian ones, but found that substitution did not work all of the times. This is the reason we switched to a different generating definition.

Code can be generated via

```
$ python -m torchsh.symbolic.codegen --help
```

which requires all `dev-requirements.txt` to be installed. To run the unit tests call

```
$ pytest
```

## References

The basic idea for using `sympy` to generate code for RSH functions is taken from https://nvlabs.github.io/instant-ngp, where it is used to generate Cuda code for forward and backward passes.
