###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-element flop costs for pointwise and reduction ops.

Fused Triton kernels, standalone ATen perf models, and the model-explorer
roofline all price an op from this one table.

A flop is one add, subtract, multiply, compare, min, or max. Ops a GPU cannot
finish in one such instruction use the normalized weights of the Livermore
Fortran Kernels (F. H. McMahon, LLNL UCRL-53745, 1986), which scale each op to
the execution time of an add: divide and square root are 4, exp, log, and the
other elementary functions are 8. A composite op costs the sum of its formula.
Reductions are priced per input element.
"""

# Constants
_ARITH = 1
_DIVIDE = 4
_ELEMENTARY = 8

# 1 / (1 + exp(-x)): negate, exp, add, divide.
_SIGMOID = _ARITH + _ELEMENTARY + _ARITH + _DIVIDE

OP_FLOPS_PER_ELEM: dict[str, int] = {
    # Data movement only.
    "_to_copy": 0,
    "to": 0,
    "copy": 0,
    "copy_": 0,
    "clone": 0,
    "embedding": 0,
    "pad": 0,
    # One arithmetic, compare, or select instruction.
    "add": _ARITH,
    "sub": _ARITH,
    "mul": _ARITH,
    "neg": _ARITH,
    "abs": _ARITH,
    "sign": _ARITH,
    "square": _ARITH,
    "relu": _ARITH,
    "maximum": _ARITH,
    "minimum": _ARITH,
    "clamp_min": _ARITH,
    "clamp_max": _ARITH,
    "eq": _ARITH,
    "ne": _ARITH,
    "lt": _ARITH,
    "le": _ARITH,
    "gt": _ARITH,
    "ge": _ARITH,
    "threshold_backward": _ARITH,
    "clamp": 2 * _ARITH,
    # pow(x, 2) is x * x. Graph labels and fused kernel names do not record the
    # exponent, and the square in RMSNorm and variance is the common case.
    # Traced aten::pow reads its exponent through pow_flops_per_elem.
    "pow": _ARITH,
    # Divide and square-root class.
    "div": _DIVIDE,
    "reciprocal": _DIVIDE,
    "sqrt": _DIVIDE,
    "rsqrt": _DIVIDE,
    "floor_divide": _DIVIDE + _ARITH,
    # Elementary functions.
    "exp": _ELEMENTARY,
    "log": _ELEMENTARY,
    "log1p": _ELEMENTARY,
    "tanh": _ELEMENTARY,
    "erf": _ELEMENTARY,
    "sin": _ELEMENTARY,
    "cos": _ELEMENTARY,
    # Composites.
    "sigmoid": _SIGMOID,
    # x / (1 + exp(-x)), the same terms as sigmoid.
    "silu": _SIGMOID,
    # x * 0.5 * (1 + erf(x / sqrt(2))).
    "gelu": 4 * _ARITH + _ELEMENTARY,
    # gelu(approximate="tanh"):
    # 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
    "gelu_tanh": 8 * _ARITH + _ELEMENTARY,
    # log1p(exp(x)).
    "softplus": 2 * _ELEMENTARY,
    # grad * y * (1 - y).
    "sigmoid_backward": 3 * _ARITH,
    # Reductions, per input element.
    "sum": _ARITH,
    "mean": _ARITH,
    "prod": _ARITH,
    "cumsum": _ARITH,
    "cumprod": _ARITH,
    "max": _ARITH,
    "min": _ARITH,
    "amax": _ARITH,
    "amin": _ARITH,
    "argmax": _ARITH,
    "argmin": _ARITH,
    "any": _ARITH,
    "all": _ARITH,
    # |x|^p accumulated; p = 2 squares, p = 1 and inf take abs.
    "norm": 2 * _ARITH,
    "linalg_norm": 2 * _ARITH,
    # sum(x), then sum((x - mean)^2).
    "var": 4 * _ARITH,
    "std": 4 * _ARITH,
    # max, x - max, exp, sum.
    "logsumexp": 3 * _ARITH + _ELEMENTARY,
    # max, x - max, exp, sum, scale by 1 / sum.
    "softmax": 4 * _ARITH + _ELEMENTARY,
    "_softmax": 4 * _ARITH + _ELEMENTARY,
}

# Exponents PyTorch's pow kernel replaces with cheaper math. 0 fills ones and
# 1 copies the input.
_POW_SCALAR_EXPONENT_FLOPS = {
    0.0: 0,
    1.0: 0,
    2.0: _ARITH,
    3.0: 2 * _ARITH,
    0.5: _DIVIDE,
    -0.5: _DIVIDE,
    -1.0: _DIVIDE,
    -2.0: _ARITH + _DIVIDE,
}
# exp(y * log(x)).
_POW_GENERAL_FLOPS = _ELEMENTARY + _ARITH + _ELEMENTARY


def op_flops_per_elem(op_name: str) -> int:
    """Flops per element for ``aten::add_``, ``aten.add``, or ``add``.

    An in-place ``_`` suffix costs the same as the out-of-place op. A name the
    table does not list contributes 0, as view and layout ops do.
    """
    name = op_name.removeprefix("aten::").removeprefix("aten.")
    if name not in OP_FLOPS_PER_ELEM and name.endswith("_"):
        name = name[:-1]
    return OP_FLOPS_PER_ELEM.get(name, 0)


def pow_flops_per_elem(exponent: float | None) -> int:
    """Flops per element of ``x ** exponent``; ``None`` is a tensor exponent."""
    if exponent is None:
        return _POW_GENERAL_FLOPS
    return _POW_SCALAR_EXPONENT_FLOPS.get(float(exponent), _POW_GENERAL_FLOPS)
