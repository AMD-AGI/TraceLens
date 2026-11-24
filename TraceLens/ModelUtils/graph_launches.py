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

import logging
import math
from dataclasses import dataclass, field, replace

from TraceLens.ModelUtils.parallel_plan import ParallelDegree
from TraceLens.ModelUtils.torch_runtime import DeviceFacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphModel:
    """The served model whose graphs a trace replays, and the run around it.

    ``checkpoint`` is a Hugging Face model id, a checkpoint directory, or a
    ``config.json``. ``engine`` and ``engine_version`` name the serving
    engine; an empty version is read from the trace, else the newest release.
    ``library_versions`` overrides the ref of a kernel library the engine
    checkout pins. ``settings`` are environment variables and server
    arguments (a name starting ``--``) the run is known to have set; anything
    unset is unknown, and every arm it selects stays a candidate until the
    replay decides it. The setting ``reserved_memory`` is the device memory
    the run held outside its tensors, in bytes (the trace does not record
    it). ``device`` and ``nccl_version`` are the machine as the trace records
    it.
    """

    checkpoint: str
    parallel: ParallelDegree = field(default_factory=ParallelDegree)
    num_cus: int = 0
    gfx_arch: str = ""
    engine: str = ""
    engine_version: str = ""
    library_versions: tuple[tuple[str, str], ...] = ()
    settings: tuple[tuple[str, str], ...] = ()
    device: DeviceFacts | None = None
    nccl_version: str = ""
    # ``(op name, input dims)`` of ops the trace ran outside any graph: what
    # the run computed from machine state the trace does not record (its KV
    # pool size) shows in them.
    eager_shapes: tuple[tuple[str, tuple], ...] = ()
    # ``(file, function)`` of every Python frame the trace records: which
    # arms of the engine's source the run took.
    traced_functions: frozenset[tuple[str, str]] = frozenset()
    # Each distinct graph replay of the trace: the kernels it ran, by name,
    # and the batch size and token count of the step that ran it.
    replay_graphs: frozenset[tuple[tuple[str, ...], tuple[int, ...]]] = frozenset()
    # The batch sizes and token counts the trace's steps ran at.
    replay_sizes: frozenset[int] = frozenset()
    # ``(op, kernel name)`` of each kernel an op launched outside a graph.
    eager_kernels: frozenset[tuple[str, str]] = frozenset()

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
    # The Python calls it was made within that were handed tensors, innermost
    # first: ``(qualified name, dims passed, types passed, positions passed
    # None, names of the trailing keyword arguments)``.
    callers: tuple = ()


def engine_run_for(model: GraphModel):
    """The engine startup that captures ``model``'s graphs, or None if unknown."""
    from TraceLens.ModelUtils.engine_run import RECIPES, EngineRun

    if model.engine not in RECIPES or model.device is None:
        return None
    settings = dict(model.settings)
    reserved = int(settings.pop("reserved_memory", 0) or 0)
    server_args: list[str] = []
    environment = []
    for name, value in settings.items():
        if name.startswith("--"):
            server_args.extend([name, value] if value != "" else [name])
        else:
            environment.append((name, value))
    world_size = 1
    for axis in ("tp", "pp"):
        world_size *= max(model.degree(axis), 1)
    return EngineRun(
        engine=model.engine,
        release=model.engine_version,
        model=model.checkpoint,
        gpu_target=model.gfx_arch,
        device=replace(model.device, reserved_memory=reserved),
        world_size=world_size,
        nccl_version=model.nccl_version,
        server_args=tuple(server_args),
        settings=tuple(environment),
        library_overrides=model.library_versions,
        capture_sizes=tuple(sorted(model.replay_sizes)),
    )


# Engine run -> what its startup captured, or the error that stopped it.
_RUNS: dict = {}
# The device memory a run holds outside its tensors is found by trying
# reservations: the first step, and how often to refine.
_RESERVATION_STEP = 1 << 30
_CALIBRATION_ROUNDS = 4
# A dimension the reservation moves is this large, and differs from the
# trace's by at most this fraction.
_LARGE_SIZE = 1 << 16
# A machine with other memory held (other processes, a smaller static
# fraction) can hold a few times fewer rows: within a factor of four.
_MEMORY_DIMENSION_SPREAD = 0.75
_CALIBRATED: dict = {}


def memory_dimension(graphs, eager_shapes) -> tuple | None:
    """``((op, argument, axis), predicted, traced)`` of a size set by free memory.

    The trace's eager ops show the run's real sizes. A captured launch of
    the same op whose arguments agree but for one large size, within a few
    times of it, shows a size the machine's free memory decided (a KV pool's
    rows).
    """
    from collections import Counter

    observed: dict[str, list[tuple]] = {}
    for name, dims in eager_shapes:
        observed.setdefault(name, []).append(dims)
    votes: Counter = Counter()
    for graph in graphs:
        for launch in graph.launches:
            for traced in observed.get(launch.op_name, ()):
                found = _one_size_apart(tuple(launch.record.dims), traced)
                if found is not None:
                    dims = tuple(
                        tuple(d) if isinstance(d, list | tuple) else d
                        for d in launch.record.dims
                    )
                    # The launch is told apart by its other sizes: the op's
                    # other launches (a request pool's rows) are not it.
                    votes[
                        (launch.op_name, *found, _masked(dims, found[0], found[1]))
                    ] += 1
    if not votes:
        return None
    (name, argument, axis, predicted, traced, masked), _ = votes.most_common(1)[0]
    return (name, argument, axis, masked), predicted, traced


def _one_size_apart(predicted: tuple, traced: tuple):
    """``(argument, axis, predicted, traced)``: one large size of one argument, close.

    Token counts differ between a graph and the eager calls; a size the
    machine's free memory decides is large (a pool's rows) and close.
    """
    for argument, (mine, theirs) in enumerate(zip(predicted, traced)):
        mine, theirs = tuple(mine or ()), tuple(theirs or ())
        if mine == theirs or len(mine) != len(theirs) or not mine:
            continue
        if not all(isinstance(size, int) for size in (*mine, *theirs)):
            continue
        axes = [axis for axis, (a, b) in enumerate(zip(mine, theirs)) if a != b]
        if len(axes) != 1:
            continue
        a, b = mine[axes[0]], theirs[axes[0]]
        if min(a, b) >= _LARGE_SIZE and abs(a - b) <= _MEMORY_DIMENSION_SPREAD * max(
            a, b
        ):
            return argument, axes[0], a, b
    return None


def _size_at(graphs, where, near: int | None = None) -> int | None:
    """The size ``where`` names in ``graphs``: the first launch's, or the one nearest ``near``.

    Launches of one op with the same other sizes (a request pool's rows and a
    token pool's) are told apart by how close they are to the size followed.
    """
    if near is not None:
        sizes = _sizes_at(graphs, where)
        return min(
            sizes,
            key=lambda size: abs(math.log(max(size, 1) / max(near, 1))),
            default=None,
        )
    found = _sizes_at(graphs, where)
    return found[0] if found else None


def _sizes_at(graphs, where) -> list[int]:
    found = []
    name, argument, axis, *masked = where
    for graph in graphs:
        for launch in graph.launches:
            if launch.op_name == name and argument < len(launch.record.dims):
                if masked:
                    mine = tuple(
                        tuple(d) if isinstance(d, list | tuple) else d
                        for d in launch.record.dims
                    )
                    if len(mine) != len(masked[0]) or not isinstance(
                        mine[argument], tuple
                    ):
                        continue
                    if (
                        len(mine[argument]) <= axis
                        or _masked(mine, argument, axis) != masked[0]
                    ):
                        continue
                dims = launch.record.dims[argument]
                if (
                    isinstance(dims, tuple | list)
                    and axis < len(dims)
                    and isinstance(dims[axis], int)
                ):
                    found.append(dims[axis])
    return found


def setting_dimensions(graphs, eager_shapes) -> list[tuple]:
    """``((op, argument, axis), predicted, traced, settings)`` of sizes a setting set.

    A captured launch whose arguments are an eager call's of the same op but
    for one size, where the trace's calls all agree on that size, shows a
    size the run's settings decided otherwise. ``settings`` are those the
    launch's arguments were computed from, most often seen first. Sizes
    that vary between the trace's calls, or between graphs captured for
    different batches, are a batch's or a token count's, not a setting's.
    """
    from collections import Counter

    observed: dict[str, list[tuple]] = {}
    for name, dims in eager_shapes:
        observed.setdefault(name, []).append(dims)
    found: dict[tuple, list] = {}
    for graph in graphs:
        for launch in graph.launches:
            mine = tuple(
                tuple(dims) if isinstance(dims, list | tuple) else dims
                for dims in launch.record.dims
            )
            for traced in observed.get(launch.op_name, ()):
                apart = _one_axis_apart(mine, traced)
                if apart is None:
                    continue
                argument, axis, predicted, size = apart
                where = (launch.op_name, argument, axis, _masked(mine, argument, axis))
                entry = found.setdefault(where, [set(), set(), Counter()])
                entry[0].add(predicted)
                entry[1].add(size)
                entry[2].update(getattr(launch, "shaped_by", ()))
    # The size every launch the same but for it takes, in every captured graph.
    across: dict[tuple, set] = {where: set() for where in found}
    for graph in graphs:
        for launch in graph.launches:
            mine = tuple(
                tuple(dims) if isinstance(dims, list | tuple) else dims
                for dims in launch.record.dims
            )
            for (name, argument, axis, masked), sizes in across.items():
                if (
                    name == launch.op_name
                    and len(mine) == len(masked)
                    and isinstance(mine[argument], tuple)
                    and len(mine[argument]) > axis
                    and _masked(mine, argument, axis) == masked
                ):
                    sizes.add(mine[argument][axis])
    return [
        (
            where,
            next(iter(predicted)),
            next(iter(traced)),
            [name for name, _ in settings.most_common()],
        )
        for where, (predicted, traced, settings) in sorted(
            found.items(), key=lambda item: -sum(item[1][2].values())
        )
        if len(predicted) == 1
        and len(traced) == 1
        and settings
        and across[where] == predicted
    ]


def _masked(dims: tuple, argument: int, axis: int) -> tuple:
    """``dims`` with one size left open wherever it is: the launch, whatever that size is.

    A size one setting decides is often in several arguments (a pool's keys
    and its values); each stays open with it.
    """
    size = dims[argument][axis]
    return tuple(
        (
            tuple(None if item == size else item for item in shape)
            if isinstance(shape, tuple)
            else shape
        )
        for shape in dims
    )


def _one_axis_apart(predicted: tuple, traced: tuple):
    """``(argument, axis, predicted, traced)``: all arguments equal but one size."""
    if len(predicted) != len(traced):
        return None
    found = None
    for argument, (mine, theirs) in enumerate(zip(predicted, traced)):
        mine, theirs = tuple(mine or ()), tuple(theirs or ())
        if mine == theirs:
            continue
        if found is not None or len(mine) != len(theirs) or not mine:
            return None
        axes = [axis for axis, (a, b) in enumerate(zip(mine, theirs)) if a != b]
        if len(axes) != 1 or not all(
            isinstance(size, int) for size in (*mine, *theirs)
        ):
            return None
        found = (argument, axes[0], mine[axes[0]], theirs[axes[0]])
    return found


def calibrated_run(run, eager_shapes):
    """``run`` holding the device memory that makes its free-memory sizes the trace's.

    The trace's eager calls show the run's real sizes. A size the machine's
    free memory decided (a KV pool's rows) is fitted through the memory the
    run held outside its tensors (the runtime, library workspaces), which
    the trace does not record either.
    """
    key = (run, tuple(eager_shapes))
    if key in _CALIBRATED:
        return _CALIBRATED[key]
    _CALIBRATED[key] = found = _calibrate(run, eager_shapes)
    return found


# The least share of a replay's kernels a graph that ran it launches.
_LAUNCH_SHARE = 0.75

# How many sizes are fitted through settings, at most, and how many settings
# are tried for each.
_FITTED_SIZES = 4
_FITTING_CANDIDATES = 4


def _fitted_settings(run, eager_shapes):
    """``run`` with the numeric settings that make its setting-decided sizes the trace's."""
    from collections import Counter

    fitted: set[tuple] = set()
    for _ in range(_FITTED_SIZES):
        # A start that stopped keeps the launches it made: enough to size.
        started = _started(run)
        if started is None or not eager_shapes:
            return run
        pending = [
            found
            for found in setting_dimensions(started.graphs, eager_shapes)
            if found[0] not in fitted
        ]
        if not pending:
            return run
        where, predicted, traced, settings = pending[0]
        fitted.add(where)
        known = _known_settings(run)
        # A setting that shapes nearly every launch (the parallel size) says
        # little about this one size: the most specific few are tried.
        shaping = Counter(
            name
            for graph in started.graphs
            for launch in graph.launches
            for name in set(launch.shaped_by)
        )
        ranked = sorted(
            (name for name in settings if name not in known),
            key=lambda name: (shaping[name], settings.index(name)),
        )
        lines = {}
        for name in ranked:
            line = _trial_line(started, name, traced)
            if line is not None:
                lines[name] = line
            if len(lines) == _FITTING_CANDIDATES:
                break
        # Every candidate's two trials at once: a size moves with one of them.
        _start_together(
            [
                _with_settings(run, {name: str(value)})
                for name, line in lines.items()
                for value in line
            ]
        )
        for name, line in lines.items():
            better = _fit_setting(run, name, where, traced, line)
            if better is not None:
                logger.info(
                    "%s = %s sizes %s as the trace's %s",
                    name,
                    _known_settings(better).get(name),
                    where,
                    traced,
                )
                run = better
                break
    return run


def _numeric(value) -> bool:
    """Whether a variable's value is unset or a number."""
    if value is None:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def _trial_line(started, name, traced):
    """Two values of ``name`` to see a size move along, or None if it is no number."""
    destination = next(
        (dest for dest, (option, _) in started.options.items() if option == name), None
    )
    if destination is not None:
        if destination not in started.option_numbers:
            return None
        default = started.option_defaults.get(destination)
    else:
        from TraceLens.ModelUtils.engine_settings import number_variables

        default = started.settings_read.get(name)
        if not (
            default is not None
            and _numeric(default)
            or name in number_variables(started.roots)
        ):
            return None
    if isinstance(default, float):
        return (default, round(default * 0.9, 6))
    return (traced, 2 * traced)


def _fit_setting(run, name, where, traced, line):
    """``run`` with ``name`` set so the size at ``where`` is ``traced``, or None.

    The size is taken to move linearly with the setting: two trials give
    the line, and the value on it is tried.
    """
    first, second = line

    def size_with(value):
        trial = _with_settings(run, {name: str(value)})
        captured = _started(trial)
        return trial, (
            _size_at(captured.graphs, where) if captured is not None else None
        )

    trial, at_first = size_with(first)
    if at_first == traced:
        return trial
    _, at_second = size_with(second)
    if at_first is None or at_second is None or at_first == at_second:
        return None
    value = first + (traced - at_first) * (second - first) / (at_second - at_first)
    value = round(value) if isinstance(first, int) else round(value, 6)
    trial, found = size_with(value)
    return trial if found == traced else None


def _calibrate(run, eager_shapes):
    """``run`` holding the device memory that makes its free-memory sizes the trace's.

    Try a reservation, see how far the size moves, and solve, as the size
    falls near-linearly as the reservation grows.
    """

    def sized(reserved):
        trial = replace(run, device=replace(run.device, reserved_memory=int(reserved)))
        captured = _captured(trial)
        return trial, (captured.graphs if captured is not None else None)

    _start_together(
        [
            replace(run, device=replace(run.device, reserved_memory=int(reserved)))
            for reserved in (
                run.device.reserved_memory,
                run.device.reserved_memory + _RESERVATION_STEP,
            )
        ]
        if eager_shapes
        else []
    )
    trial, graphs = sized(run.device.reserved_memory)
    if not graphs or not eager_shapes:
        return trial
    found = memory_dimension(graphs, eager_shapes)
    if found is None:
        return trial
    where, predicted, target = found
    if predicted == target:
        return trial
    reserved = run.device.reserved_memory
    _, moved = sized(reserved + _RESERVATION_STEP)
    shifted = _size_at(moved or (), where, near=predicted)
    if shifted is None or shifted == predicted:
        logger.warning("the reservation does not move %s; sizes stay as run", where)
        return trial
    per_byte = (predicted - shifted) / _RESERVATION_STEP
    current = predicted
    for _ in range(_CALIBRATION_ROUNDS):
        reserved += (current - target) / per_byte
        trial, graphs = sized(round(reserved))
        current = _size_at(graphs or (), where, near=current)
        if current is None or current == target:
            break
    if current != target:
        logger.warning(
            "calibrating %s reached %s, the trace has %s", where, current, target
        )
    return trial


# Starts tried together run side by side, at most this many at once.
_PARALLEL_STARTS = 6


def _share_cores(workers: int, levels: dict | None = None) -> None:
    """A start run side by side computes on its share of the cores, not all of them,
    and logs as the process that started it does (``levels``: logger -> level)."""
    import os

    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // (4 * workers)))
    if levels:
        logging.basicConfig(
            level=levels.get("", logging.WARNING),
            format="%(asctime)s %(name)s %(message)s",
        )
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def _logging_levels() -> dict:
    """This process's root logging level and this package's loggers' levels."""
    levels = {"": logging.root.level}
    for name, logger in logging.root.manager.loggerDict.items():
        if (
            isinstance(logger, logging.Logger)
            and name.startswith("TraceLens")
            and logger.level
        ):
            levels[name] = logger.level
    return levels


def _start_together(runs) -> None:
    """Start ``runs`` side by side, each in its own process; each caches itself.

    A start is one process interpreting the engine: trials that do not
    depend on each other need not wait for each other.
    """
    import concurrent.futures
    import multiprocessing

    from TraceLens.ModelUtils.engine_run import engine_graphs, is_cached

    pending = []
    for run in runs:
        if run not in _RUNS and run not in pending and not is_cached(run):
            pending.append(run)
    if len(pending) < 2 or multiprocessing.parent_process() is not None:
        # One at a time when only one waits, or inside a start run side by
        # side (whose program imported this one's main module again).
        return
    logger.info(
        "%d starts side by side: %s",
        len(pending),
        "; ".join(_described(run) for run in pending),
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(pending), _PARALLEL_STARTS),
        mp_context=context,
        initializer=_share_cores,
        initargs=(min(len(pending), _PARALLEL_STARTS), _logging_levels()),
    ) as pool:
        for future in [pool.submit(engine_graphs, run) for run in pending]:
            try:
                future.result()
            except (
                Exception
            ) as error:  # noqa: BLE001 - the start reports it again when read
                logger.warning("a start run side by side failed: %s", error)


def _started(run):
    """What ``run``'s startup captured and reached, or None if it cannot run."""
    from TraceLens.ModelUtils.engine_run import EngineRunError, engine_graphs

    if run not in _RUNS:
        logger.info("start %s", _described(run))
        try:
            _RUNS[run] = engine_graphs(run)
        except EngineRunError as error:
            logger.warning("graph shapes not inferred: %s", error)
            _RUNS[run] = None
    return _RUNS[run]


def _described(run) -> str:
    """The settings ``run`` states, with its reservation and capture sizes, for the log."""
    stated = " ".join(
        f"{name}={value}" if value else name
        for name, value in sorted(_known_settings(run).items())
    )
    return f"[{stated} reserved={run.device.reserved_memory} sizes={list(run.capture_sizes)}]"


def _captured(run):
    started = _started(run)
    if started is None or started.error:
        if started is not None:
            logger.warning("graph shapes not inferred: %s", started.error)
        return None
    return started


# A setting inferred from a run may open code that says what another was:
# runs are repeated until no new setting is found, at most this many times.
_SETTLING_ROUNDS = 6
_SETTLED: dict = {}
# The replay's kernels decide a setting in at most this many rounds, each
# trying at most this many settings' values.
_SEARCH_ROUNDS = 6
_TRIALS_PER_ROUND = 8
_BATCHES_WITHOUT_GAIN = 3


def _known_settings(run) -> dict[str, str]:
    """The run's environment variables and ``--options``, with their values.

    The options naming the model and its parallelism are known too: the
    recipe that starts the engine states them.
    """
    from TraceLens.ModelUtils.engine_run import MODEL_ARGUMENTS

    known = dict(run.settings)
    arguments = [
        *MODEL_ARGUMENTS.get(run.engine, lambda *_: [])(run.model, run.world_size),
        *run.server_args,
    ]
    for index, argument in enumerate(arguments):
        if argument.startswith("--"):
            following = arguments[index + 1] if index + 1 < len(arguments) else ""
            known[argument] = "" if following.startswith("--") else following
    return known


def _with_settings(run, found: dict[str, str]):
    server_args = list(run.server_args)
    environment = list(run.settings)
    for name, value in sorted(found.items()):
        if name.startswith("--"):
            server_args.extend([name, value] if value != "" else [name])
        else:
            environment.append((name, value))
    return replace(run, server_args=tuple(server_args), settings=tuple(environment))


def _framed_run(run, traced_functions):
    """``run`` with the settings the trace's Python frames show, to a fixed point.

    A function the trace shows that the run did not reach was called behind
    a setting the run decided the other way (:mod:`.engine_settings`). Each
    setting found is added and the run repeated, as the arm it opens may
    reach code that decides another.
    """
    from TraceLens.ModelUtils.engine_settings import inferred_settings

    for _ in range(_SETTLING_ROUNDS):
        started = _started(run)
        if started is None or not traced_functions:
            return run
        found = inferred_settings(
            started.roots,
            traced_functions,
            started.executed,
            _known_settings(run),
            started.options,
        )
        if not found:
            return run
        logger.info("the trace shows settings the run did not state: %s", found)
        run = _with_settings(run, found)
    return run


def _replay_pairs(started, replay_graphs):
    """``[(replayed kernel names, captured graph or None)]``.

    A replay ran the graph captured for its step's batch, or the next size
    up (padding); None when the start captured no graph it could have run.
    """
    pairs = []
    for kernels, sizes in replay_graphs:
        # A graph launches about as many kernels as its replay runs (a split
        # reduction adds a few); a decode graph of one request is not the
        # prefill of one request, twice its kernels.
        padded = [
            (padding, graph)
            for graph in started.graphs
            if _runs_at(graph, sizes)
            and len(graph.launches) >= _LAUNCH_SHARE * len(kernels)
            and (padding := _padding(graph, sizes)) is not None
        ]
        least = min((padding for padding, _ in padded), default=None)
        candidates = [graph for padding, graph in padded if padding == least]
        pairs.append((kernels, candidates))
    return pairs


def _runs_at(graph, sizes) -> bool:
    """Whether ``graph`` was captured for one of ``sizes`` (a step's batch or its
    token count), padded at most to twice it (a batch runs the next captured
    size up). A graph is captured for one of them: a decode graph for the
    batch, whose tokens may be several per request."""
    integers = {
        integer
        for _, passed in graph.context
        for integer in passed
        if isinstance(integer, int)
    }
    return any(size <= integer <= 2 * size for size in sizes for integer in integers)


def _replayed_launches(started, replay_graphs, eager_kernels=frozenset()):
    """``(named launches explaining a replay, named launches it never ran, replays no graph ran)``.

    Each replay is judged against the graph captured for its own size; of
    several captured for it, the one whose named launches cover the most of
    its kernels and name the fewest it did not run.
    """
    pairs = _replay_pairs(started, replay_graphs)
    matches = _symbol_matches(started, replay_graphs)
    # An op whose launch names no kernel launched, outside a graph, the
    # kernels the trace shows it launching: one of those it explains too.
    launched_by: dict[str, set] = {}
    for op, kernel in eager_kernels:
        launched_by.setdefault(op, set()).add(kernel)
    explaining, wrong, uncovered = 0, [], 0
    for kernels, candidates in pairs:
        names = set(kernels)
        best, best_score = None, None
        for graph in candidates:
            launches = [launch for launch in graph.launches if launch.kernels]
            reached = set().union(
                *(matches[kernel] for launch in launches for kernel in launch.kernels)
            )
            reached |= names & set().union(
                *(
                    launched_by.get(launch.op_name, set())
                    for launch in graph.launches
                    if not launch.kernels
                )
            )
            # A graph the replay ran explains most of its kernels; one captured
            # for its batch but of another step (a decode graph for a prefill
            # of one request) does not.
            if len(names & reached) * 2 < len(
                names & (_named(matches) | set().union(*launched_by.values()))
            ):
                continue
            unrun = [
                launch
                for launch in launches
                if not any(matches[k] & names for k in launch.kernels)
            ]
            explained = _kernels_explained(kernels, graph, matches, launched_by)
            score = (explained, -len(unrun))
            if best_score is None or score > best_score:
                best, best_score = (explained, unrun), score
        if best is None:
            uncovered += 1
            continue
        explaining += best[0]
        wrong.extend(best[1])
    return explaining, wrong, uncovered


def _kernels_explained(kernels, graph, matches, launched_by) -> int:
    """How many of a replay's kernels ``graph``'s launches account for, counting repeats.

    A kernel name the replay ran several times needs as many launches that
    can make it: one launch more of an op already launched elsewhere in the
    graph (a gather the draft adds) is one kernel more explained.
    """
    from collections import Counter

    ran = Counter(kernels)
    able: Counter = Counter()
    for launch in graph.launches:
        if launch.kernels:
            made = set().union(*(matches[kernel] for kernel in launch.kernels))
        else:
            made = launched_by.get(launch.op_name, set())
        able.update(made & ran.keys())
    return sum(min(count, able[name]) for name, count in ran.items())


def _named(matches) -> set:
    """The replayed kernel names some launch of the run names."""
    return set().union(*matches.values()) if matches else set()


def _symbol_matches(started, replay_graphs):
    """The replayed kernel names each launched symbol is."""
    from TraceLens.ModelUtils.kernel_alignment import kernel_is

    names = {name for kernels, _ in replay_graphs for name in kernels}
    symbols = {
        kernel
        for graph in started.graphs
        for launch in graph.launches
        for kernel in launch.kernels
    }
    return {
        symbol: frozenset(name for name in names if kernel_is(name, (symbol,)))
        for symbol in symbols
    }


def _fit(started, replay_graphs, eager_kernels=frozenset()) -> tuple:
    """How well a run's launches explain the replays' kernels.

    ``(-replays no captured graph ran, replayed kernels a launch names,
    -launches naming a kernel their replay never ran, whether it started,
    how much of the engine it ran)``: the run captured every graph the
    replay ran and took its arms and no others; failing that, it got
    further. A startup that stopped keeps the launches it made.
    """
    explained, wrong, uncovered = _replayed_launches(
        started, replay_graphs, eager_kernels
    )
    return -uncovered, explained, -len(wrong), not started.error, len(started.executed)


def _gain(fit: tuple) -> tuple:
    """A fit without its count of launches the replay never ran.

    How much of the engine a start ran says how far a failing start got;
    between starts that both started it says nothing of the replay.
    """
    return fit[:2] + ((fit[3],) if fit[3] else fit[3:])


def _deciding_settings(started, replay_graphs, known) -> list[str]:
    """Settings that decided the launches the replay never ran, likeliest first.

    Each launch knows the settings that decided the branches it was made
    within. A setting is likelier the more launches the replay never ran it
    decided, and the fewer of those it did run: one that decided every
    launch (the model's dtype) tells nothing apart. A startup that stopped
    short offers those that decided where it stopped, nearest first.
    """
    from collections import Counter

    _, wrong, _ = _replayed_launches(started, replay_graphs)
    judged = [
        launch
        for _, graphs in _replay_pairs(started, replay_graphs)
        for graph in graphs[:1]
        for launch in graph.launches
        if launch.kernels
    ]
    wrong_ids = {id(launch) for launch in wrong}
    against: Counter = Counter()
    right: Counter = Counter()
    nearness: Counter = Counter()
    for launch in judged:
        if id(launch) in wrong_ids:
            against.update(set(launch.control))
            unknown = [name for name in reversed(launch.control) if name not in known]
            for depth, name in enumerate(dict.fromkeys(unknown)):
                nearness[name] += 1.0 / (1 + depth)
        else:
            right.update(set(launch.control))
    # Nearer the launch, and deciding more of what the replay never ran
    # than of what it did.
    votes: dict[str, float] = {
        name: nearness[name]
        / max(against[name], 1)
        * against[name]
        / (against[name] + right[name])
        for name in against
        if name not in known
    }
    if started.error:
        for depth, name in enumerate(reversed(started.error_control)):
            if name not in known:
                votes[name] = votes.get(name, 0.0) + 1.0 / (1 + depth)
    return sorted(votes, key=lambda name: (-votes[name], name))


def _capture_settings(started, replay_graphs, run, known) -> list[str]:
    """Settings that decide which graphs a start captures, when some replay has none.

    A replay no captured graph covers ran a graph the start never made; the
    engine's own recipes for the model name the graph settings it is run
    with (``--cuda-graph-...``, ``..._GRAPH_...``).
    """
    from TraceLens.ModelUtils.engine_settings import recipe_values
    from TraceLens.ModelUtils.engine_sources import engine_sources

    if _replayed_launches(started, replay_graphs)[2] == 0:
        return []
    sources = engine_sources(
        run.engine, run.gpu_target, run.release, dict(run.library_overrides)
    )
    declared = {option for option, _ in started.options.values()}
    return [
        name
        for name in sorted(recipe_values(sources.engine_checkout, run.model))
        if "graph" in name.lower()
        and name not in known
        and (name in declared or name in started.settings_read)
    ]


def _checkpoint_settings(started, known) -> list[str]:
    """Checkpoint entries the run's loaders look for and the run has not stated.

    The weight files hold no tensor, so whether the checkpoint holds an
    entry that makes model state of its own is open; the replay's kernels
    decide it like any setting.
    """
    from TraceLens.ModelUtils.engine_settings import checkpoint_entries

    return [name for name in checkpoint_entries(started.executed) if name not in known]


def _switch_like(value) -> bool:
    """Whether an environment variable's value (None when unset) reads as a switch."""
    return value is None or str(value).strip().lower() in (
        "",
        "0",
        "1",
        "true",
        "false",
        "on",
        "off",
        "yes",
        "no",
    )


def _setting_values(name, run, started, model) -> list[str]:
    """Values to try for ``name``: the engine's recipes', then its source's."""
    from TraceLens.ModelUtils.engine_settings import (
        CHECKPOINT_ENTRY,
        compared_values,
        recipe_values,
    )

    if name.startswith(CHECKPOINT_ENTRY):
        # The checkpoint holds the entry (the value is its length, which the
        # trace may not show; a caller may state it).
        return ["1"]
    from TraceLens.ModelUtils.engine_sources import engine_sources

    sources = engine_sources(
        run.engine, run.gpu_target, run.release, dict(run.library_overrides)
    )
    recipes = recipe_values(sources.engine_checkout, run.model)
    if name.startswith("--"):
        destination = next(
            (dest for dest, (option, _) in started.options.items() if option == name),
            None,
        )
        switch = destination is not None and started.options[destination][1]
        if switch:
            return [""]
        compared = compared_values(started.roots).get(destination or "", set())
    else:
        compared = compared_values(started.roots).get(name, set())
    values = [value for value in sorted(recipes.get(name, ())) if value]
    values += [value for value in sorted(compared) if value not in values]
    if not name.startswith("--"):
        from TraceLens.ModelUtils.engine_settings import (
            number_variables,
            switch_variables,
        )

        current = started.settings_read.get(name)
        read_as = switch_variables(started.roots) | number_variables(started.roots)
        if (
            current is not None
            and _switch_like(current)
            or current is None
            and name in read_as
        ):
            # A switch, or a number: 1, then 0. A variable holding a path or a
            # name is neither.
            values += [value for value in ("1", "0") if value not in values]
    else:
        choices = started.option_choices.get(destination or "")
        if choices is not None:
            # An option that names the values it takes takes no others.
            values = [value for value in values if value in choices]
        default = started.option_defaults.get(destination or "")
        values = [value for value in values if value != str(default)]
    current = started.settings_read.get(name)
    return [value for value in values if value != current]


def _searched_run(run, model):
    """``run`` with settings that make its launches the replay's kernels.

    The replay's kernel names decide what the trace leaves unstated: a
    setting that decided a launch the replay never ran is tried at the
    values the engine's recipes and source give it, and kept when the run
    then explains more of the replay and launches less it never ran.
    """
    replay = model.replay_graphs
    if not replay:
        return run
    started = _started(run)
    if started is None:
        return run
    fit = _fit(started, replay, model.eager_kernels)
    for _ in range(_SEARCH_ROUNDS):
        known = _known_settings(run)
        # What the checkpoint holds first: few, and each decides one thing.
        # Then what decides the graphs captured, when a replay ran one the
        # start never captured: that counts before any kernel explained.
        names = _checkpoint_settings(started, known)
        names += [
            name
            for name in _capture_settings(started, replay, run, known)
            if name not in names
        ]
        names += [
            name
            for name in _deciding_settings(started, replay, known)
            if name not in names
        ]
        # Each setting's likeliest value first, then the next values.
        values = {name: _setting_values(name, run, started, model) for name in names}
        depth = max((len(found) for found in values.values()), default=0)
        ordered = [
            _with_settings(run, {name: values[name][level]})
            for level in range(depth)
            for name in values
            if level < len(values[name])
        ]
        improved = None
        # A batch that explains nothing more passes to the next, a few times;
        # once every replay has its graph, the likeliest batch decides.
        batches = _BATCHES_WITHOUT_GAIN if fit[0] < 0 else 1
        for batch in range(batches):
            trials = ordered[
                batch * _TRIALS_PER_ROUND : (batch + 1) * _TRIALS_PER_ROUND
            ]
            if not trials:
                break
            _start_together(trials)
            for trial in trials:
                tried = _started(trial)
                if tried is None:
                    continue
                trial_fit = _fit(tried, replay, model.eager_kernels)
                logger.info(
                    "trial %s: fit %s against %s", _described(trial), trial_fit, fit
                )
                # A setting is kept for what it makes the run explain (or how
                # much further it gets), not for launches it merely drops.
                if _gain(trial_fit) <= _gain(fit):
                    continue
                if improved is None or trial_fit > improved[2]:
                    improved = (trial, tried, trial_fit)
            if improved is not None:
                break
        if improved is None:
            return run
        run, started, fit = improved
        logger.info("the replay's kernels decide %s", _known_settings(run))
    return run


def settled_run(run, model):
    """``run`` with what its trace shows of the settings it had.

    The trace's Python frames, its eager calls' sizes, and its replayed
    kernels each decide settings the trace does not state; a setting one
    decides can open code that lets another decide one more, so all are
    repeated until none adds any.
    """
    for _ in range(_SETTLING_ROUNDS):
        settled = _framed_run(run, model.traced_functions)
        # What code runs first, then the sizes it makes: a size the wrong
        # code makes (a draft head the target's width) is no setting's.
        settled = _searched_run(settled, model)
        settled = _fitted_settings(settled, model.eager_shapes)
        if settled == run:
            return run
        run = settled
    return run


def _launch(recorded) -> Launch:
    record = recorded.record
    return Launch(
        op_name=recorded.op_name,
        input_shapes=tuple(
            tuple(dims) if isinstance(dims, list | tuple) else ()
            for dims in record.dims
        ),
        input_types=tuple(record.types),
        concrete=tuple(record.concrete),
        input_strides=tuple(
            tuple(strides) if isinstance(strides, list | tuple) else ()
            for strides in record.strides
        ),
        kernels=tuple(recorded.kernels),
        same_call=recorded.same_call,
        callers=tuple(
            (
                caller.name,
                caller.record.dims,
                caller.record.types,
                caller.nones,
                caller.keywords,
            )
            for caller in recorded.callers
        ),
    )


def _padding(graph, sizes) -> int | None:
    """How far ``graph`` was captured above the replay's batch or token count.

    A graph is captured within a call handed its batch size (or token
    count); a batch with no graph of its own is padded to the next captured
    size up.
    """
    above = [
        integer - size
        for _, integers in graph.context
        for integer in integers
        for size in sizes
        if isinstance(integer, int) and integer >= size
    ]
    return min(above) if above else None


def select_graph(
    graphs, invocation: Invocation, kernel_names: list[str], op_families=None
):
    """The captured graph a replay ran: aligning best, captured nearest its batch.

    The replayed graph pairs the most replayed kernels with its launches,
    named kernels first, less the launches it would have made that the
    replay did not run. Graphs of several sizes launch the same kernels; the
    one replayed was captured for the batch, or the least padding above it.
    """
    from TraceLens.ModelUtils.kernel_alignment import align_kernels

    sizes = (invocation.batch_size, invocation.tokens)
    best, best_score = None, None
    for graph in graphs:
        launches = [_launch(recorded) for recorded in graph.launches]
        assigned = align_kernels(kernel_names, launches, op_families)
        paired = [launch for launch in assigned if launch is not None]
        used = {id(launch) for launch in paired}
        unused = sum(1 for launch in launches if id(launch) not in used)
        padding = _padding(graph, sizes)
        # A graph explains the kernels it pairs, and is the wrong graph by
        # every launch it would have made that the replay did not run.
        score = (
            sum(1 for launch in paired if launch.kernels),
            len(paired) - unused,
            padding is not None,
            -(padding or 0),
        )
        if best_score is None or score > best_score:
            best, best_score = launches, score
    return best


def graph_launches(
    model: GraphModel,
    invocation: Invocation,
    kernel_names: list[str],
    op_families: dict[str, set[str]] | None = None,
) -> list[Launch] | None:
    """The launches one replay of ``invocation`` issues, in order.

    ``kernel_names`` are the kernels the replay ran. They decide every arm of
    the forward that a setting the run left unstated would otherwise keep
    open. ``op_families`` are the kernel families each op launched outside
    any graph, as the trace shows them. ``None`` means the served model's
    source could not be read for this graph, and its kernels stay launch
    rows without shapes.
    """
    run = engine_run_for(model)
    if run is None:
        return None
    key = (run, model.traced_functions, model.replay_graphs, model.eager_kernels)
    if key not in _SETTLED:
        _SETTLED[key] = settled_run(run, model)
    run = _SETTLED[key]
    if not any(name == "reserved_memory" for name, _ in model.settings):
        run = calibrated_run(run, model.eager_shapes)
    captured = _captured(run)
    if captured is None or not captured.graphs:
        return None
    return select_graph(captured.graphs, invocation, kernel_names, op_families)
