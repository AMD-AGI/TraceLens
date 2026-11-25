###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An inference engine's own startup, run from source on the traced machine.

The engine is checked out at the release the trace ran, with the kernel
libraries its build pins (:mod:`.engine_sources`), and interpreted
(:mod:`.source_interpreter`) on a model of the traced machine
(:mod:`.run_machine`). Its GPU is meta tensors that keep the run's device
memory and addresses (:mod:`.torch_runtime`); its compiled libraries are read
from their C++ source (:mod:`.native_index`, :mod:`.native_host`); its
weight files are listed, never read (:mod:`.checkpoint_index`). One rank starts
as the engine starts it, and every graph the engine captures on the way is
kept with the launches it recorded.

The only engine knowledge here is how a rank is started (:data:`RECIPES`).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pickle
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from TraceLens.ModelUtils.torch_runtime import CapturedGraph, DeviceFacts

logger = logging.getLogger(__name__)

CACHE = Path.home() / ".cache" / "tracelens" / "engine_runs"
# Python distributions the run uses from the host process as they are: pure
# libraries whose behaviour does not depend on the machine.
HOST_PACKAGES = frozenset(
    {
        "torch",
        "numpy",
        "transformers",
        "huggingface_hub",
        "packaging",
        "typing_extensions",
        "einops",
        "regex",
        "pandas",
        "jinja2",
        "markupsafe",
    }
)
_REPOSITORY_ID = re.compile(r"^[\w.\-]+/[\w.\-]+$")


class EngineRunError(RuntimeError):
    """The engine's startup could not be followed to its end."""


@dataclass(frozen=True)
class EngineRun:
    """One rank of a served model, as the trace describes its run.

    ``model`` is a Hugging Face repository id or a local directory.
    ``server_args`` are the engine's command-line arguments beyond the model
    and its parallelism; a value naming a Hugging Face repository (a draft
    model) is read like the model. ``settings`` are environment variables.
    ``capture_sizes`` are the batch sizes and token counts the trace's
    graphs ran: the engine captures only the graphs those run in (each size
    padded up to one it captures), and sizes them as it would have anyway.
    """

    engine: str
    release: str
    model: str
    gpu_target: str
    device: DeviceFacts
    world_size: int = 1
    nccl_version: str = ""
    server_args: tuple[str, ...] = ()
    settings: tuple[tuple[str, str], ...] = ()
    library_overrides: tuple[tuple[str, str], ...] = ()
    capture_sizes: tuple[int, ...] = ()

    def key(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:24]


@dataclass
class EngineGraphs:
    """What one rank's startup captured."""

    graphs: list[CapturedGraph] = field(default_factory=list)
    # Environment variables the engine read, and the value each had (None
    # when unset): what an unstated setting could change.
    settings_read: dict[str, str | None] = field(default_factory=dict)
    # Libraries whose import needs what their checkout does not hold.
    unmodelled: dict[str, str] = field(default_factory=dict)
    # ``(file, function)`` the run executed, and the command-line options the
    # engine declares (destination -> option, switch): what tells which
    # settings would have taken another way.
    executed: set[tuple[str, str]] = field(default_factory=set)
    options: dict[str, tuple[str, bool]] = field(default_factory=dict)
    option_defaults: dict[str, object] = field(default_factory=dict)
    option_choices: dict[str, tuple] = field(default_factory=dict)
    option_numbers: set[str] = field(default_factory=set)
    # The source directories the run read, engine and libraries.
    roots: tuple[str, ...] = ()
    # Why the rank did not finish starting; empty when it did. And the
    # settings that decided the branches the error was raised within.
    error: str = ""
    error_control: tuple[str, ...] = ()


def is_cached(run: EngineRun, cache: Path = CACHE) -> bool:
    """Whether a start of ``run`` is cached."""
    return (cache / f"{run.engine}-{run.key()}-{_model_version()}.pkl").is_file()


def engine_graphs(run: EngineRun, cache: Path = CACHE) -> EngineGraphs:
    """Every graph ``run``'s rank captures at startup (cached on disk).

    A startup that stops keeps what it captured and reached, with its
    ``error``; an engine with no recipe raises :class:`EngineRunError`.
    """
    path = cache / f"{run.engine}-{run.key()}-{_model_version()}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    result = _start(run)
    _write_cache(path, result)
    return result


def _write_cache(path: Path, value) -> None:
    """Cache ``value`` at ``path`` whole: a process reading it meanwhile (a
    start run side by side) finds it absent, never half written."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(pickle.dumps(value))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# The modules that model the run: a startup cached under other versions of
# them is not this one's.
_MODELLING_MODULES = (
    "checkpoint_index",
    "engine_run",
    "library_models",
    "native_host",
    "native_index",
    "run_machine",
    "sandbox_fs",
    "source_interpreter",
    "torch_runtime",
)


@functools.lru_cache(maxsize=1)
def _model_version() -> str:
    """The modelling code's version, as this process loaded it."""
    digest = hashlib.sha256()
    for name in _MODELLING_MODULES:
        digest.update((Path(__file__).parent / f"{name}.py").read_bytes())
    return digest.hexdigest()[:12]


class _Ready(BaseException):
    """The rank reported itself ready: its startup is complete."""


class _ReadyPipe:
    """The pipe a rank reports readiness on, to whoever started it."""

    def send(self, *args, **kwargs):
        raise _Ready

    def close(self):
        pass


def _sglang(interpreter, argv: list[str], capture_sizes=()) -> None:
    """Start sglang's rank 0 scheduler as its launcher does."""
    server_args_module = interpreter.import_module("sglang.srt.server_args")
    server_args = server_args_module.prepare_server_args(list(argv))
    captured = _sglang_capture_lists(interpreter, server_args, capture_sizes)
    if captured:
        server_args = server_args_module.prepare_server_args([*argv, *captured])
    port_args = server_args_module.PortArgs.init_new(server_args)
    scheduler = interpreter.import_module("sglang.srt.managers.scheduler")
    scheduler.run_scheduler_process(server_args, port_args, 0, 0, 0, None, _ReadyPipe())


def _sglang_capture_lists(interpreter, server_args, sizes) -> list[str]:
    """Options that capture only the graphs sizes ran in, sized as before.

    sglang resolves its capture lists when a rank publishes its arguments.
    It sizes each phase's graph buffers by the largest size it captures:
    that one stays, with the smallest captured size at or above each size a
    graph ran at (a batch runs padded up to a captured size).
    """
    if not sizes:
        return []
    context = interpreter.import_module("sglang.srt.runtime_context")
    context.publish(
        server_args,
        role="scheduler",
        ranks=context.SpawnRanks(world_rank=0, dp_rank=None, gpu_id=0),
    )
    config = context.get_exec().graph.cuda_graph_config
    options = []
    for phase, option in (
        ("decode", "--cuda-graph-bs-decode"),
        ("prefill", "--cuda-graph-bs-prefill"),
    ):
        phase_config = getattr(config, phase, None)
        if (
            phase_config is None
            or "disabled" in str(getattr(phase_config, "backend", "")).lower()
        ):
            continue
        listed = sorted(getattr(phase_config, "bs", None) or [])
        if not listed:
            continue
        kept = {listed[-1]}
        for size in sizes:
            above = [captured for captured in listed if captured >= size]
            if above:
                kept.add(above[0])
        if len(kept) < len(listed):
            options += [option, *(str(size) for size in sorted(kept))]
    return options


# Engine -> how one of its ranks is started, given its command line and the
# sizes its graphs are run at.
RECIPES = {"sglang": _sglang}
# Engine -> the command-line arguments naming the model and its parallelism.
MODEL_ARGUMENTS = {
    "sglang": lambda model, world_size: [
        "--model-path",
        model,
        "--tp-size",
        str(world_size),
    ]
}


def _snapshot(repository: str) -> Path:
    """A local directory with the repository's small files (config, tokenizer, code)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repository,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.txt",
                "*.py",
                "*.tiktoken",
                "*.jinja",
            ],
        )
    )


def _is_repository(value: str) -> bool:
    if not _REPOSITORY_ID.match(value) or Path(value).exists():
        return False
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(value)
    except Exception:  # noqa: BLE001 - not a model repository
        return False
    return True


def _logging_state():
    """The host process's logging configuration: the root's, and each logger's."""
    import logging

    loggers = {
        name: (logger.level, logger.disabled, logger.propagate, list(logger.handlers))
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    return logging.root.level, list(logging.root.handlers), loggers


def _restore_logging(state) -> None:
    import logging

    level, handlers, loggers = state
    logging.root.setLevel(level)
    logging.root.handlers[:] = handlers
    for name, (level, disabled, propagate, own) in loggers.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.disabled = disabled
        logger.propagate = propagate
        logger.handlers[:] = own


def _start(run: EngineRun) -> EngineGraphs:
    from TraceLens.ModelUtils.checkpoint_index import checkpoint_files
    from TraceLens.ModelUtils.engine_settings import CHECKPOINT_ENTRY
    from TraceLens.ModelUtils.engine_sources import engine_sources
    from TraceLens.ModelUtils.library_models import library_modules
    from TraceLens.ModelUtils.native_host import HostEvaluator
    from TraceLens.ModelUtils.run_machine import RunMachine
    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter
    from TraceLens.ModelUtils.torch_runtime import TorchRuntime, _holds_tensor

    if run.engine not in RECIPES:
        raise EngineRunError(f"no recipe starts a {run.engine} rank")
    sources = engine_sources(
        run.engine, run.gpu_target, run.release, dict(run.library_overrides)
    )
    installed = sources.installed
    gpu = image_gpu_target(run.gpu_target, sources.environment) or run.device.gfx_target
    device = replace(
        run.device, gfx_target=gpu, count=max(run.device.count, run.world_size)
    )
    runtime = TorchRuntime(
        device, world_size=run.world_size, triton_version=installed.get("triton", "")
    )
    machine = RunMachine(device, device.hip_version, run.nccl_version or None)
    checkpoints: dict[str, object] = {}

    # Entries a loader looks for that the run states the checkpoint holds:
    # (source file name, loader function, entry name, its length). The
    # setting's value is the entry's length (the weights are not read).
    entries = [
        (
            *name[len(CHECKPOINT_ENTRY) :].split(":", 2),
            int(value) if str(value).isdigit() else 1,
        )
        for name, value in run.settings
        if name.startswith(CHECKPOINT_ENTRY)
    ]

    def checkpoint_for(path):
        try:
            file = checkpoints.get(os.path.normpath(os.fspath(path)))
        except TypeError:
            return None
        if file is None or not entries:
            return file
        # Read from within a loader that looks for an entry: the file holds
        # it, as a placeholder of its stated length (its values decide no shape).
        # A frame names its function qualified (``Class.method``).
        loading = {
            (Path(location.file).name, location.function.rpartition(".")[2])
            for location in interpreter.stack
        }
        held = {
            entry: length
            for source, function, entry, length in entries
            if (source, function) in loading
        }
        if not held:
            return file
        logger.info("weight file %s holds the stated entries %s", file.path, held)
        import torch

        return replace(
            file,
            tensors={
                entry: torch.zeros(length, dtype=torch.int64)
                for entry, length in held.items()
            },
        )

    environment = Environment(
        settings={
            **sources.environment,
            **{
                name: value
                for name, value in run.settings
                if not name.startswith(CHECKPOINT_ENTRY)
            },
        },
        readable=(sources.engine_checkout.resolve(),),
    )
    native_factory = {}
    interpreter = SourceInterpreter(
        dict(sources.roots),
        environment,
        host_modules={
            **runtime.modules,
            **library_modules(
                lambda name: native_factory["make"](name), checkpoint_for
            ),
        },
        host_packages=HOST_PACKAGES,
    )
    machine.install(interpreter)
    index = _native_index(sources.engine_checkout)
    host = HostEvaluator(index.host_functions, interpreter)
    indexed = {sources.engine_checkout}
    # Each checkout's own declarations: two libraries may bind one name.
    own_indexes = {
        sources.engine_checkout.resolve(): _native_index(sources.engine_checkout)
    }

    def resolve(top):
        """A library the run imports, fetched on demand, its C++ indexed with it."""
        root = sources.resolve(top)
        for checkout in list(sources.checkouts):
            if checkout not in indexed:
                indexed.add(checkout)
                library = _native_index(checkout)
                own_indexes[Path(checkout).resolve()] = library
                # The host evaluator reads the same table, so it sees them too.
                index.merge(library)
                interpreter.declare_extensions(library.extensions)
                runtime.register_native_ops(library, host)
        return root

    interpreter.resolver = resolve

    def make_native(name):
        module = runtime.native_module(
            name,
            _scoped_index(name),
            host,
            interpreter.pins.get(name.partition(".")[0]),
        )
        module.machine_functions = machine.c_functions()
        return module

    def _scoped_index(name):
        """The index, with the declarations of the library ``name`` belongs to first."""
        import collections
        import copy

        package = interpreter.roots.get(name.partition(".")[0])
        own = next(
            (
                library
                for checkout, library in own_indexes.items()
                if package is not None
                and Path(package).resolve().is_relative_to(checkout)
            ),
            None,
        )
        if own is None:
            return index
        scoped = copy.copy(index)
        scoped.signatures = collections.ChainMap(own.signatures, index.signatures)
        # What a function calls, and the kernels it can launch, resolve within
        # its own library: a name another library also defines is not its.
        scoped.named = own.named
        scoped.kernels = own.kernels
        return scoped

    native_factory["make"] = make_native
    interpreter.native_module_factory = make_native
    interpreter.declare_extensions(index.extensions)
    runtime.register_native_ops(index, host)
    runtime.generated_sources = lambda: environment.files
    runtime.checkpoint_for = checkpoint_for
    runtime.log.label = interpreter.where
    runtime.log.caller = lambda: interpreter.caller_holding(_holds_tensor)
    runtime.log.callers = lambda: interpreter.callers_holding(_holds_tensor)
    runtime.log.context = interpreter.call_path
    runtime.log.control = interpreter.control_settings
    runtime.log.shaping = interpreter.argument_settings

    def local_model(value: str) -> str:
        """``value`` as the engine reads it: a repository becomes its local snapshot."""
        directory = _snapshot(value)
        environment.readable = (
            *environment.readable,
            directory.parent.parent.resolve(),
        )
        files = checkpoint_files(value)
        for relative, file in files.items():
            path = os.path.normpath(str(directory / relative))
            checkpoints[path] = file
            environment.files[path] = file
        return str(directory)

    model = local_model(run.model) if not Path(run.model).exists() else run.model
    argv = [
        *MODEL_ARGUMENTS[run.engine](model, run.world_size),
        *(
            local_model(value) if _is_repository(value) else value
            for value in run.server_args
        ),
    ]
    failure = ""
    _report_where_on_signal(interpreter)
    logging_state = _logging_state()
    try:
        with interpreter, runtime:
            RECIPES[run.engine](interpreter, argv, run.capture_sizes)
        error, where = interpreter.last_error or (None, "")
        failure = f"{type(error).__name__}: {error} at {where}"
    except _Ready:
        pass
    except KeyboardInterrupt:
        raise
    except (
        BaseException
    ) as error:  # noqa: BLE001 - the engine's own failure (its argument parser exits), reported
        failure = (
            f"{type(error).__name__}: {error} at {getattr(error, '_where', None) or ''}"
        )
    finally:
        # The engine configures logging for its own process; this one's stays as it was.
        _restore_logging(logging_state)
    graphs = list(runtime.log.graphs)
    if failure and runtime.log.eager:
        # Where it stopped: the launches made since the last capture began.
        graphs.append(CapturedGraph("<stopped>", list(runtime.log.eager)))
    return EngineGraphs(
        graphs=graphs,
        settings_read=dict(environment.reads),
        unmodelled=dict(interpreter.unmodelled),
        executed=set(interpreter.executed),
        options=dict(environment.options),
        option_defaults=dict(environment.option_defaults),
        option_choices=dict(environment.option_choices),
        option_numbers=set(environment.option_numbers),
        roots=tuple(
            sorted(
                {
                    str((Path(root) / top).resolve())
                    for top, root in interpreter.roots.items()
                    if (Path(root) / top).is_dir()
                }
            )
        ),
        error=f"{run.engine} rank did not start: {failure}" if failure else "",
        error_control=interpreter.last_error_control if failure else (),
    )


def _report_where_on_signal(interpreter) -> None:
    """``kill -USR1`` a long start to log where in the engine it is."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(
        signal.SIGUSR1,
        lambda *_: logger.warning("the start is at %s", interpreter.where()),
    )


def image_gpu_target(prefix: str, environment: dict[str, str]) -> str | None:
    """The image's own GPU target completing ``prefix`` (``gfx95`` -> ``gfx950``).

    An image builds for a list of targets (``PYTORCH_ROCM_ARCH``); the one of
    them in the traced device's family is the run's.
    """
    if not prefix.startswith("gfx"):
        return None
    targets = {
        target
        for value in environment.values()
        for target in re.findall(r"\bgfx[0-9a-f]+\b", value)
        if target.startswith(prefix)
    }
    return targets.pop() if len(targets) == 1 else None


def _native_index(root: Path):
    """The C/C++ declarations of one checkout (cached on disk)."""
    from TraceLens.ModelUtils.native_index import build_native_index

    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]
    path = CACHE / f"native-{key}-{_model_version()}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    index = build_native_index([root], build_roots=[root])
    _write_cache(path, index)
    return index
