###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The device launches a replayed graph runs, read from the served model's code.

A graph replay records each kernel and nothing about the call that issued it.
The served model is the checkpoint plus the inference engine that ran it, and
both are source code. Reading that code at the versions the run used gives
every launch of one forward in order, with the tensors it is called with.

Nothing here names a model. A model, an engine and a kernel library are
inputs; what they launch is whatever their source says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from TraceLens.ModelUtils.parallel_plan import ParallelDegree


@dataclass(frozen=True)
class GraphModel:
    """The served model whose graphs a trace replays, and the run around it.

    ``checkpoint`` is a Hugging Face model id, a checkpoint directory, or a
    ``config.json``. ``engine`` and ``engine_version`` name the serving
    engine; an empty version is read from the trace, else the newest release.
    ``library_versions`` overrides the ref of a kernel library the engine
    checkout pins. ``settings`` are environment variables and server
    arguments the run is known to have set; anything unset is unknown, and
    every arm it selects stays a candidate until the replay decides it.
    """

    checkpoint: str
    parallel: ParallelDegree = field(default_factory=ParallelDegree)
    num_cus: int = 0
    gfx_arch: str = ""
    engine: str = ""
    engine_version: str = ""
    library_versions: tuple[tuple[str, str], ...] = ()
    settings: tuple[tuple[str, str], ...] = ()

    @property
    def tp(self) -> int:
        return self.parallel.size("tp")

    def degree(self, name: str) -> int:
        """The size of one parallelism axis (``tp``, ``moe_ep``, ``pp``, ...)."""
        return self.parallel.size(name)


@dataclass(frozen=True)
class Invocation:
    """One model run. ``tokens`` is the M dimension of the graph's GEMMs.

    ``scope`` is every annotation enclosing the graph launch, innermost
    last: the engine writes them, and its source says which runner each one
    belongs to.
    """

    batch_size: int
    seq_len: int
    past_len: int
    tokens: int
    prefill: bool = False
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class Launch:
    """One device launch in the graph, in the order the forward issues it.

    ``kernels`` are the device symbols the issuing call can launch, as its
    source defines them (a Triton ``@jit`` function, a ``__global__``
    function of the compiled module). Empty means the source does not name
    them, and alignment places the launch by order between named ones.
    """

    op_name: str
    input_shapes: tuple[tuple[int, ...], ...]
    # Dtypes in argument order, as the profiler records them on the captured
    # call. The roofline reads these, not only the shapes.
    input_types: tuple[str, ...] = ()
    # Scalar arguments, aligned with ``input_shapes``. ``""`` is a tensor.
    concrete: tuple[str, ...] = ()
    # Stride of each argument. Empty means the tensor is contiguous.
    input_strides: tuple[tuple[int, ...], ...] = ()
    kernels: tuple[str, ...] = ()
    layer: int | None = None
    # This kernel is launched by the same CPU call as the previous launch.
    same_call: bool = False


def graph_launches(
    model: GraphModel, invocation: Invocation, kernel_names: list[str]
) -> list[Launch] | None:
    """The launches one replay of ``invocation`` issues, in order.

    ``kernel_names`` are the kernels the replay ran. They decide every arm of
    the forward that a setting the run left unstated would otherwise keep
    open. ``None`` means the served model's source could not be read for
    this graph, and its kernels stay launch rows without shapes.
    """
    return None
