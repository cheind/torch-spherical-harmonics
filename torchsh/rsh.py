"""Real spherical harmonics in Cartesian form for PyTorch.

This is an autogenerated file. See
https://github.com/cheind/torch-spherical-harmonics
for more information.
"""

import torch


def rsh_cart_0(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 0.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,1) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
        ],
        -1,
    )


def rsh_cart_1(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 1.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,4) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
        ],
        -1,
    )


def rsh_cart_2(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 2.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,9) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
        ],
        -1,
    )


def rsh_cart_3(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 3.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,16) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
        ],
        -1,
    )


def rsh_cart_4(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 4.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,25) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
            2.5033429417967 * xy * (x2 - y2),
            1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.0892062058076386 * yz * (52.5 * z2 - 22.5),
            -3.17356640745613 * z2 + 3.70249414203215 * z4 + 0.317356640745613,
            0.0892062058076386 * xz * (52.5 * z2 - 22.5),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
        ],
        -1,
    )


def rsh_cart_5(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 5.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,36) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
            2.5033429417967 * xy * (x2 - y2),
            1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.0892062058076386 * yz * (52.5 * z2 - 22.5),
            -3.17356640745613 * z2 + 3.70249414203215 * z4 + 0.317356640745613,
            0.0892062058076386 * xz * (52.5 * z2 - 22.5),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            7.00140860629515 * y * (-0.9375 * x2 * y2 + 0.46875 * x4 + 0.09375 * y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (3.0 * x2 - y2) * (472.5 * z2 - 52.5),
            0.0913054625709205 * xy * z * (157.5 * z2 - 52.5),
            0.241571547304372 * y * (-26.25 * z2 + 39.375 * z4 + 1.875),
            1.87120515925478 * z * (-4.375 * z2 + 3.9375 * z4 + 0.9375),
            0.241571547304372 * x * (-26.25 * z2 + 39.375 * z4 + 1.875),
            0.0456527312854602 * z * (x2 - y2) * (157.5 * z2 - 52.5),
            0.00931882475114763 * x * (x2 - 3.0 * y2) * (472.5 * z2 - 52.5),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            7.00140860629515 * x * (-0.9375 * x2 * y2 + 0.09375 * x4 + 0.46875 * y4),
        ],
        -1,
    )


def rsh_cart_6(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 6.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,49) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
            2.5033429417967 * xy * (x2 - y2),
            1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.0892062058076386 * yz * (52.5 * z2 - 22.5),
            -3.17356640745613 * z2 + 3.70249414203215 * z4 + 0.317356640745613,
            0.0892062058076386 * xz * (52.5 * z2 - 22.5),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            7.00140860629515 * y * (-0.9375 * x2 * y2 + 0.46875 * x4 + 0.09375 * y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (3.0 * x2 - y2) * (472.5 * z2 - 52.5),
            0.0913054625709205 * xy * z * (157.5 * z2 - 52.5),
            0.241571547304372 * y * (-26.25 * z2 + 39.375 * z4 + 1.875),
            1.87120515925478 * z * (-4.375 * z2 + 3.9375 * z4 + 0.9375),
            0.241571547304372 * x * (-26.25 * z2 + 39.375 * z4 + 1.875),
            0.0456527312854602 * z * (x2 - y2) * (157.5 * z2 - 52.5),
            0.00931882475114763 * x * (x2 - 3.0 * y2) * (472.5 * z2 - 52.5),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            7.00140860629515 * x * (-0.9375 * x2 * y2 + 0.09375 * x4 + 0.46875 * y4),
            43.7237827322825 * xy * (-0.3125 * x2 * y2 + 0.09375 * x4 + 0.09375 * y4),
            2.36661916223175 * yz * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00427144889505798 * xy * (x2 - y2) * (5197.5 * z2 - 472.5),
            0.00584892228263444 * yz * (3.0 * x2 - y2) * (1732.5 * z2 - 472.5),
            0.0701870673916132 * xy * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.221950995245231 * yz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            14.6844857238222 * z2**3
            + 6.67476623810098 * z2
            - 20.024298714303 * z4
            - 0.317846011338142,
            0.221950995245231 * xz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            0.0350935336958066 * (x2 - y2) * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.00584892228263444 * xz * (x2 - 3.0 * y2) * (1732.5 * z2 - 472.5),
            0.0010678622237645 * (5197.5 * z2 - 472.5) * (-6.0 * x2 * y2 + x4 + y4),
            2.36661916223175 * xz * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            0.683184105191914 * x2**3
            + 10.2477615778787 * x2 * y4
            - 10.2477615778787 * x4 * y2
            - 0.683184105191914 * y2**3,
        ],
        -1,
    )


def rsh_cart_7(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 7.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,64) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
            2.5033429417967 * xy * (x2 - y2),
            1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.0892062058076386 * yz * (52.5 * z2 - 22.5),
            -3.17356640745613 * z2 + 3.70249414203215 * z4 + 0.317356640745613,
            0.0892062058076386 * xz * (52.5 * z2 - 22.5),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            7.00140860629515 * y * (-0.9375 * x2 * y2 + 0.46875 * x4 + 0.09375 * y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (3.0 * x2 - y2) * (472.5 * z2 - 52.5),
            0.0913054625709205 * xy * z * (157.5 * z2 - 52.5),
            0.241571547304372 * y * (-26.25 * z2 + 39.375 * z4 + 1.875),
            1.87120515925478 * z * (-4.375 * z2 + 3.9375 * z4 + 0.9375),
            0.241571547304372 * x * (-26.25 * z2 + 39.375 * z4 + 1.875),
            0.0456527312854602 * z * (x2 - y2) * (157.5 * z2 - 52.5),
            0.00931882475114763 * x * (x2 - 3.0 * y2) * (472.5 * z2 - 52.5),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            7.00140860629515 * x * (-0.9375 * x2 * y2 + 0.09375 * x4 + 0.46875 * y4),
            43.7237827322825 * xy * (-0.3125 * x2 * y2 + 0.09375 * x4 + 0.09375 * y4),
            2.36661916223175 * yz * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00427144889505798 * xy * (x2 - y2) * (5197.5 * z2 - 472.5),
            0.00584892228263444 * yz * (3.0 * x2 - y2) * (1732.5 * z2 - 472.5),
            0.0701870673916132 * xy * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.221950995245231 * yz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            14.6844857238222 * z2**3
            + 6.67476623810098 * z2
            - 20.024298714303 * z4
            - 0.317846011338142,
            0.221950995245231 * xz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            0.0350935336958066 * (x2 - y2) * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.00584892228263444 * xz * (x2 - 3.0 * y2) * (1732.5 * z2 - 472.5),
            0.0010678622237645 * (5197.5 * z2 - 472.5) * (-6.0 * x2 * y2 + x4 + y4),
            2.36661916223175 * xz * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            0.683184105191914 * x2**3
            + 10.2477615778787 * x2 * y4
            - 10.2477615778787 * x4 * y2
            - 0.683184105191914 * y2**3,
            15.0861382938581
            * y
            * (
                0.328125 * x2**3
                + 0.984375 * x2 * y4
                - 1.640625 * x4 * y2
                - 0.046875 * y2**3
            ),
            5.2919213236038 * xy * z * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            9.98394571852353e-5
            * y
            * (67567.5 * z2 - 5197.5)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00239614697244565 * xy * z * (x2 - y2) * (22522.5 * z2 - 5197.5),
            0.00397356022507413
            * y
            * (3.0 * x2 - y2)
            * (-2598.75 * z2 + 5630.625 * z4 + 118.125),
            0.0561946276120613 * xy * z * (-866.25 * z2 + 1126.125 * z4 + 118.125),
            0.206472245902897
            * y
            * (187.6875 * z2**3 + 59.0625 * z2 - 216.5625 * z4 - 2.1875),
            2.18509686118416
            * z
            * (13.40625 * z2**3 + 9.84375 * z2 - 21.65625 * z4 - 1.09375),
            0.206472245902897
            * x
            * (187.6875 * z2**3 + 59.0625 * z2 - 216.5625 * z4 - 2.1875),
            0.0280973138060306
            * z
            * (x2 - y2)
            * (-866.25 * z2 + 1126.125 * z4 + 118.125),
            0.00397356022507413
            * x
            * (x2 - 3.0 * y2)
            * (-2598.75 * z2 + 5630.625 * z4 + 118.125),
            0.000599036743111412
            * z
            * (22522.5 * z2 - 5197.5)
            * (-6.0 * x2 * y2 + x4 + y4),
            9.98394571852353e-5
            * x
            * (67567.5 * z2 - 5197.5)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            2.6459606618019 * z * (x2**3 + 15.0 * x2 * y4 - 15.0 * x4 * y2 - y2**3),
            15.0861382938581
            * x
            * (
                0.046875 * x2**3
                + 1.640625 * x2 * y4
                - 0.984375 * x4 * y2
                - 0.328125 * y2**3
            ),
        ],
        -1,
    )


def rsh_cart_8(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 8.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,81) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            0.48860251190292 * y,
            0.48860251190292 * z,
            0.48860251190292 * x,
            1.09254843059208 * xy,
            1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            4.72034871941315 * y * (0.375 * x2 - 0.125 * y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (7.5 * z2 - 1.5),
            1.49270533036046 * z * (1.25 * z2 - 0.75),
            0.304697199642977 * x * (7.5 * z2 - 1.5),
            1.44530572132028 * z * (x2 - y2),
            4.72034871941315 * x * (0.125 * x2 - 0.375 * y2),
            2.5033429417967 * xy * (x2 - y2),
            1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.0892062058076386 * yz * (52.5 * z2 - 22.5),
            -3.17356640745613 * z2 + 3.70249414203215 * z4 + 0.317356640745613,
            0.0892062058076386 * xz * (52.5 * z2 - 22.5),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            7.00140860629515 * y * (-0.9375 * x2 * y2 + 0.46875 * x4 + 0.09375 * y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (3.0 * x2 - y2) * (472.5 * z2 - 52.5),
            0.0913054625709205 * xy * z * (157.5 * z2 - 52.5),
            0.241571547304372 * y * (-26.25 * z2 + 39.375 * z4 + 1.875),
            1.87120515925478 * z * (-4.375 * z2 + 3.9375 * z4 + 0.9375),
            0.241571547304372 * x * (-26.25 * z2 + 39.375 * z4 + 1.875),
            0.0456527312854602 * z * (x2 - y2) * (157.5 * z2 - 52.5),
            0.00931882475114763 * x * (x2 - 3.0 * y2) * (472.5 * z2 - 52.5),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            7.00140860629515 * x * (-0.9375 * x2 * y2 + 0.09375 * x4 + 0.46875 * y4),
            43.7237827322825 * xy * (-0.3125 * x2 * y2 + 0.09375 * x4 + 0.09375 * y4),
            2.36661916223175 * yz * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00427144889505798 * xy * (x2 - y2) * (5197.5 * z2 - 472.5),
            0.00584892228263444 * yz * (3.0 * x2 - y2) * (1732.5 * z2 - 472.5),
            0.0701870673916132 * xy * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.221950995245231 * yz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            14.6844857238222 * z2**3
            + 6.67476623810098 * z2
            - 20.024298714303 * z4
            - 0.317846011338142,
            0.221950995245231 * xz * (-78.75 * z2 + 86.625 * z4 + 13.125),
            0.0350935336958066 * (x2 - y2) * (-236.25 * z2 + 433.125 * z4 + 13.125),
            0.00584892228263444 * xz * (x2 - 3.0 * y2) * (1732.5 * z2 - 472.5),
            0.0010678622237645 * (5197.5 * z2 - 472.5) * (-6.0 * x2 * y2 + x4 + y4),
            2.36661916223175 * xz * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            0.683184105191914 * x2**3
            + 10.2477615778787 * x2 * y4
            - 10.2477615778787 * x4 * y2
            - 0.683184105191914 * y2**3,
            15.0861382938581
            * y
            * (
                0.328125 * x2**3
                + 0.984375 * x2 * y4
                - 1.640625 * x4 * y2
                - 0.046875 * y2**3
            ),
            5.2919213236038 * xy * z * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            9.98394571852353e-5
            * y
            * (67567.5 * z2 - 5197.5)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00239614697244565 * xy * z * (x2 - y2) * (22522.5 * z2 - 5197.5),
            0.00397356022507413
            * y
            * (3.0 * x2 - y2)
            * (-2598.75 * z2 + 5630.625 * z4 + 118.125),
            0.0561946276120613 * xy * z * (-866.25 * z2 + 1126.125 * z4 + 118.125),
            0.206472245902897
            * y
            * (187.6875 * z2**3 + 59.0625 * z2 - 216.5625 * z4 - 2.1875),
            2.18509686118416
            * z
            * (13.40625 * z2**3 + 9.84375 * z2 - 21.65625 * z4 - 1.09375),
            0.206472245902897
            * x
            * (187.6875 * z2**3 + 59.0625 * z2 - 216.5625 * z4 - 2.1875),
            0.0280973138060306
            * z
            * (x2 - y2)
            * (-866.25 * z2 + 1126.125 * z4 + 118.125),
            0.00397356022507413
            * x
            * (x2 - 3.0 * y2)
            * (-2598.75 * z2 + 5630.625 * z4 + 118.125),
            0.000599036743111412
            * z
            * (22522.5 * z2 - 5197.5)
            * (-6.0 * x2 * y2 + x4 + y4),
            9.98394571852353e-5
            * x
            * (67567.5 * z2 - 5197.5)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            2.6459606618019 * z * (x2**3 + 15.0 * x2 * y4 - 15.0 * x4 * y2 - y2**3),
            15.0861382938581
            * x
            * (
                0.046875 * x2**3
                + 1.640625 * x2 * y4
                - 0.984375 * x4 * y2
                - 0.328125 * y2**3
            ),
            62.2017416682521
            * xy
            * (
                0.09375 * x2**3
                + 0.65625 * x2 * y4
                - 0.65625 * x4 * y2
                - 0.09375 * y2**3
            ),
            2.91570664069932
            * yz
            * (7.0 * x2**3 + 21.0 * x2 * y4 - 35.0 * x4 * y2 - y2**3),
            1.57570656324281e-5
            * xy
            * (1013512.5 * z2 - 67567.5)
            * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            5.10587282657803e-5
            * yz
            * (337837.5 * z2 - 67567.5)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00147275890257803
            * xy
            * (x2 - y2)
            * (-33783.75 * z2 + 84459.375 * z4 + 1299.375),
            0.0028519853513317
            * yz
            * (3.0 * x2 - y2)
            * (-11261.25 * z2 + 16891.875 * z4 + 1299.375),
            0.0463392770473559
            * xy
            * (2815.3125 * z2**3 + 649.6875 * z2 - 2815.3125 * z4 - 19.6875),
            0.193851103820053
            * yz
            * (402.1875 * z2**3 + 216.5625 * z2 - 563.0625 * z4 - 19.6875),
            -109.150287144679 * z2**3
            - 11.4493308193719 * z2
            + 58.4733681132208 * z4**2
            + 62.9713195065454 * z4
            + 0.318036967204775,
            0.193851103820053
            * xz
            * (402.1875 * z2**3 + 216.5625 * z2 - 563.0625 * z4 - 19.6875),
            0.0231696385236779
            * (x2 - y2)
            * (2815.3125 * z2**3 + 649.6875 * z2 - 2815.3125 * z4 - 19.6875),
            0.0028519853513317
            * xz
            * (x2 - 3.0 * y2)
            * (-11261.25 * z2 + 16891.875 * z4 + 1299.375),
            0.000368189725644507
            * (-33783.75 * z2 + 84459.375 * z4 + 1299.375)
            * (-6.0 * x2 * y2 + x4 + y4),
            5.10587282657803e-5
            * xz
            * (337837.5 * z2 - 67567.5)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            7.87853281621404e-6
            * (1013512.5 * z2 - 67567.5)
            * (x2**3 + 15.0 * x2 * y4 - 15.0 * x4 * y2 - y2**3),
            2.91570664069932
            * xz
            * (x2**3 + 35.0 * x2 * y4 - 21.0 * x4 * y2 - 7.0 * y2**3),
            -20.4099464848952 * x2**3 * y2
            - 20.4099464848952 * x2 * y2**3
            + 0.72892666017483 * x4**2
            + 51.0248662122381 * x4 * y4
            + 0.72892666017483 * y4**2,
        ],
        -1,
    )


__all__ = [
    "rsh_cart_0",
    "rsh_cart_1",
    "rsh_cart_2",
    "rsh_cart_3",
    "rsh_cart_4",
    "rsh_cart_5",
    "rsh_cart_6",
    "rsh_cart_7",
    "rsh_cart_8",
]
