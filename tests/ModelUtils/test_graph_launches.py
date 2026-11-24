###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A replay's launches come from the engine run its trace describes."""

import textwrap

from TraceLens.ModelUtils.engine_settings import inferred_settings
from TraceLens.ModelUtils.graph_launches import (
    GraphModel,
    Invocation,
    engine_run_for,
    memory_dimension,
    setting_dimensions,
    select_graph,
)
from TraceLens.ModelUtils.parallel_plan import ParallelDegree
from TraceLens.ModelUtils.torch_runtime import (
    ArgumentRecord,
    CapturedGraph,
    DeviceFacts,
    RecordedLaunch,
)


def _launch(op, dims, kernels=()):
    record = ArgumentRecord(
        dims,
        tuple("float" for _ in dims),
        tuple(() for _ in dims),
        tuple("" for _ in dims),
    )
    return RecordedLaunch(op, record, kernels)


def test_settings_split_into_server_arguments_environment_and_the_reservation():
    model = GraphModel(
        checkpoint="org/model",
        parallel=ParallelDegree(tp=8),
        gfx_arch="gfx95",
        engine="sglang",
        engine_version="v1",
        settings=(
            ("--enable-thing", ""),
            ("--mem", "0.85"),
            ("FLAG", "1"),
            ("reserved_memory", "1024"),
        ),
        device=DeviceFacts(name="gpu", total_memory=1 << 34),
    )
    run = engine_run_for(model)
    assert run.server_args == ("--enable-thing", "--mem", "0.85")
    assert run.settings == (("FLAG", "1"),)
    assert run.device.reserved_memory == 1024
    assert run.world_size == 8
    # No device facts, or an engine with no recipe: nothing to run.
    assert engine_run_for(GraphModel(checkpoint="org/model", engine="sglang")) is None
    assert (
        engine_run_for(
            GraphModel(checkpoint="org/model", engine="other", device=model.device)
        )
        is None
    )


def test_the_replayed_graph_is_captured_for_its_batch_and_explains_its_kernels():
    def graph(batch, launches):
        return CapturedGraph(
            "", launches, context=(("capture", ()), ("capture_one", (batch,)))
        )

    small = graph(4, [_launch("lib::norm", ((4, 8),), ("norm_kernel",))])
    large = graph(8, [_launch("lib::norm", ((8, 8),), ("norm_kernel",))])
    bigger = graph(
        8,
        [
            _launch("lib::norm", ((8, 8),), ("norm_kernel",)),
            _launch("aten::add", ((8, 8),)),
        ],
    )
    invocation = Invocation(batch_size=8, seq_len=1, past_len=0, tokens=8)
    chosen = select_graph([small, bigger, large], invocation, ["void norm_kernel<1>()"])
    # Captured for batch 8, and no launch the replay did not run.
    assert [launch.input_shapes for launch in chosen] == [((8, 8),)]
    # A batch no graph was captured for runs the next size up, padded.
    padded = Invocation(batch_size=6, seq_len=1, past_len=0, tokens=6)
    chosen = select_graph(
        [graph(16, large.launches), large, small], padded, ["void norm_kernel<1>()"]
    )
    assert [launch.input_shapes for launch in chosen] == [((8, 8),)]


def test_a_size_free_memory_decides_is_one_large_size_close_to_the_traces():
    graph = CapturedGraph("", [_launch("lib::store", ((32, 1024), (1205716, 1024)))])
    eager = (("lib::store", ((640, 1024), (1211961, 1024))), ("lib::other", ((5, 5),)))
    where = ("lib::store", 1, 0, ((32, 1024), (None, 1024)))
    assert memory_dimension([graph], eager) == (where, 1205716, 1211961)
    # Token counts differ by more than a pool's rows, and are small.
    assert (
        memory_dimension([graph], (("lib::store", ((640, 1024), (1205716, 1024))),))
        is None
    )


def test_a_setting_every_call_of_a_traced_function_needs_is_inferred(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "gate.py").write_text(textwrap.dedent("""
        import os
        from engine.environ import envs

        def use_fast():
            return envs.ENGINE_BACKEND.get() == "fast"

        def run(x):
            if use_fast():
                return fast_path(x)
            if os.environ.get("ENGINE_TRACE") == "1":
                trace_path(x)
            return slow_path(x)

        def fast_path(x):
            return x

        def trace_path(x):
            return x

        def slow_path(x):
            return x

        class Indexer:
            def __init__(self, args):
                self.use_small = args.kernel.enable_small_indexer

            def forward(self, x):
                small = self.use_small and x is not None
                if small:
                    def run_small():
                        return small_indexer(x)
                    return run_small()
                return wide_indexer(x)

        def small_indexer(x):
            return x

        def wide_indexer(x):
            return x

        def never_reached(x):
            if envs.ENGINE_OTHER.get():
                small_unreached(x)

        def small_unreached(x):
            return x
    """))
    traced = {
        ("pkg/gate.py", "run"),
        ("pkg/gate.py", "fast_path"),
        ("pkg/gate.py", "trace_path"),
    }
    executed = {("/x/pkg/gate.py", "run"), ("/x/pkg/gate.py", "slow_path")}
    assert inferred_settings(tmp_path, traced, executed, {}) == {
        "ENGINE_BACKEND": "fast",
        "ENGINE_TRACE": "1",
    }
    # A setting the caller states stays as stated.
    assert inferred_settings(
        tmp_path, traced, executed, {"ENGINE_BACKEND": "slow"}
    ) == {"ENGINE_TRACE": "1"}
    # An option read off the parsed arguments, through an attribute and a
    # local; a call from a function the trace does not show says nothing.
    traced = {
        ("pkg/gate.py", "forward"),
        ("pkg/gate.py", "small_indexer"),
        ("pkg/gate.py", "small_unreached"),
    }
    executed = {("/x/pkg/gate.py", "forward"), ("/x/pkg/gate.py", "wide_indexer")}
    options = {"enable_small_indexer": ("--enable-small-indexer", True)}
    assert inferred_settings(tmp_path, traced, executed, {}, options) == {
        "--enable-small-indexer": ""
    }


def test_a_size_a_setting_decides_is_one_the_trace_agrees_on():
    def launch(dims, shaped_by=()):
        record = ArgumentRecord(
            dims,
            tuple("int" for _ in dims),
            tuple(() for _ in dims),
            tuple("" for _ in dims),
        )
        return RecordedLaunch("aten::index", record, (), shaped_by=shaped_by)

    graph = CapturedGraph(
        "", [launch(((257, 4096), ()), ("--max-requests",)), launch(((1, 64), (64,)))]
    )
    eager = (
        ("aten::index", ((9, 4096), ())),
        ("aten::index", ((7, 64), (64,))),
        ("aten::index", ((5, 64), (64,))),
    )
    # The request pool's rows: one size apart, and every eager call agrees.
    # The token axis varies between the trace's calls: no setting's.
    rows = ("aten::index", 0, 0, ((None, 4096), ()))
    assert setting_dimensions([graph], eager) == [(rows, 257, 9, ["--max-requests"])]
    # A size that changes with the graph's batch is the batch's.
    other = CapturedGraph("", [launch(((129, 4096), ()), ("--max-requests",))])
    assert setting_dimensions([graph, other], eager) == []


def test_only_the_replayed_graph_s_own_launches_say_which_settings_went_wrong():
    from TraceLens.ModelUtils.engine_run import EngineGraphs
    from TraceLens.ModelUtils.graph_launches import _deciding_settings, _fit

    def launch(kernel, control=()):
        return RecordedLaunch(
            "op", ArgumentRecord((), (), (), ()), (kernel,), control=control
        )

    def graph(batch, launches):
        return CapturedGraph("", launches, context=(("capture_one", (batch,)),))

    small = graph(4, [launch("norm"), launch("split_gemm", ("ENGINE_GEMM",))])
    large = graph(
        8,
        [
            launch("norm"),
            launch("wide_gemm"),
            launch("slow_attention", ("ENGINE_ATTENTION",)),
        ],
    )
    started = EngineGraphs(graphs=[small, large])
    # Batch 6 ran the graph captured for 8, padded; the small graph's other
    # GEMM is right for its own size and says nothing.
    replay = frozenset({(("norm", "wide_gemm", "fast_attention"), (6, 6))})
    assert _fit(started, replay)[:4] == (0, 2, -1, True)
    assert _deciding_settings(started, replay, {}) == ["ENGINE_ATTENTION"]


def test_only_the_graphs_the_replay_ran_are_captured_sized_as_before():
    from types import SimpleNamespace

    from TraceLens.ModelUtils.engine_run import _sglang_capture_lists

    config = SimpleNamespace(
        decode=SimpleNamespace(bs=[1, 2, 4, 8, 16, 32], backend="full"),
        prefill=SimpleNamespace(bs=[4, 8, 16], backend="breakable"),
    )
    context = SimpleNamespace(
        publish=lambda *args, **kwargs: None,
        SpawnRanks=lambda **kwargs: kwargs,
        get_exec=lambda: SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=config)
        ),
    )
    interpreter = SimpleNamespace(import_module=lambda name: context)
    # Batch 3 runs padded to 4; the largest size stays, as it sizes the buffers.
    assert _sglang_capture_lists(interpreter, None, (1, 3)) == [
        "--cuda-graph-bs-decode",
        "1",
        "4",
        "32",
        "--cuda-graph-bs-prefill",
        "4",
        "16",
    ]
    assert _sglang_capture_lists(interpreter, None, ()) == []


def test_a_call_after_an_early_return_needs_the_return_not_taken(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "meta.py").write_text(textwrap.dedent("""
        import os

        def is_unified():
            return os.environ.get("ENGINE_KV") == "unified"

        def attach(meta):
            if not is_unified():
                return
            build_unified(meta)

        def build_unified(meta):
            return meta
    """))
    traced = {("pkg/meta.py", "attach"), ("pkg/meta.py", "build_unified")}
    executed = {("/x/pkg/meta.py", "attach")}
    assert inferred_settings(tmp_path, traced, executed, {}) == {"ENGINE_KV": "unified"}


def test_a_replay_no_graph_covers_asks_the_recipes_graph_settings(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from TraceLens.ModelUtils import engine_settings, engine_sources
    from TraceLens.ModelUtils.engine_run import EngineGraphs, EngineRun
    from TraceLens.ModelUtils.graph_launches import _capture_settings

    monkeypatch.setattr(
        engine_sources,
        "engine_sources",
        lambda *args, **kwargs: SimpleNamespace(engine_checkout=tmp_path),
    )
    monkeypatch.setattr(
        engine_settings,
        "recipe_values",
        lambda root, model: {
            "--graph-backend-prefill": {"breakable"},
            "--page-size": {"256"},
            "ENGINE_GRAPH_POOL": {"1"},
        },
    )
    graph = CapturedGraph(
        "",
        [RecordedLaunch("op", ArgumentRecord((), (), (), ()), ("decode_kernel",))],
        context=(("capture_one", (4,)),),
    )
    started = EngineGraphs(
        graphs=[graph],
        options={
            "graph_backend_prefill": ("--graph-backend-prefill", False),
            "page_size": ("--page-size", False),
        },
        settings_read={"ENGINE_GRAPH_POOL": None},
    )
    run = EngineRun("sglang", "v1", "org/model", "gfx95", DeviceFacts(name="gpu"))
    # The decode replay is covered: nothing to ask.
    assert (
        _capture_settings(started, frozenset({(("decode_kernel",), (4, 4))}), run, {})
        == []
    )
    # A prefill of 512 tokens ran a graph the start never captured.
    replays = frozenset(
        {(("decode_kernel",), (4, 4)), (("prefill_kernel", "other_kernel"), (1, 512))}
    )
    assert _capture_settings(started, replays, run, {}) == [
        "--graph-backend-prefill",
        "ENGINE_GRAPH_POOL",
    ]


def test_a_checkpoint_entry_a_loader_makes_state_from_is_a_setting(tmp_path):
    from TraceLens.ModelUtils.engine_settings import checkpoint_entries

    (tmp_path / "draft.py").write_text(textwrap.dedent("""
        class Draft:
            def load_weights(self, weights):
                params = dict(self.named_parameters())
                for name, loaded in weights:
                    if "d2t" in name:
                        self.hot_token_id = loaded + 1
                        continue
                    if name.endswith(".bias"):
                        continue
                    if "scale" in name:
                        params[name].copy_(loaded)
                    for other, weight in self.extra:
                        if "nope" in other:
                            self.other = weight
    """))
    path = str(tmp_path / "draft.py")
    # Only the entry whose loading makes state of the loader's own counts;
    # a loop over something other than what it was handed does not.
    assert checkpoint_entries({(path, "load_weights")}) == [
        "checkpoint:draft.py:load_weights:d2t"
    ]
    assert checkpoint_entries({(path, "forward")}) == []


def test_a_followed_size_is_the_launch_nearest_it_of_those_alike():
    from TraceLens.ModelUtils.graph_launches import _size_at

    # A request pool's rows and a token pool's: one op, alike but for the size.
    graph = CapturedGraph(
        "",
        [_launch("aten::index", ((9,), ())), _launch("aten::index", ((29186049,), ()))],
    )
    where = ("aten::index", 0, 0, ((None,), ()))
    assert _size_at([graph], where) == 9
    assert _size_at([graph], where, near=28000000) == 29186049


def test_a_size_is_fitted_through_the_most_specific_settings_only(monkeypatch):
    from types import SimpleNamespace

    from TraceLens.ModelUtils import graph_launches

    def launch(dims, shaped_by):
        record = ArgumentRecord(
            dims,
            tuple("int" for _ in dims),
            tuple(() for _ in dims),
            tuple("" for _ in dims),
        )
        return RecordedLaunch("aten::index", record, (), shaped_by=shaped_by)

    # Every launch is shaped by the parallel size; one pool by its own setting.
    everywhere = [launch(((1, 64),), ("--tp-size", f"--other-{n}")) for n in range(6)]
    pool = launch(
        ((257, 4096), ()),
        ("--tp-size", "--max-requests", *(f"--other-{n}" for n in range(6))),
    )
    started = SimpleNamespace(graphs=[CapturedGraph("", [pool, *everywhere])])
    tried = []
    monkeypatch.setattr(graph_launches, "_started", lambda run: started)
    monkeypatch.setattr(graph_launches, "_known_settings", lambda run: {})
    monkeypatch.setattr(
        graph_launches,
        "_trial_line",
        lambda started, name, traced: (traced, 2 * traced),
    )
    monkeypatch.setattr(graph_launches, "_start_together", lambda runs: None)
    monkeypatch.setattr(graph_launches, "_with_settings", lambda run, found: run)
    monkeypatch.setattr(
        graph_launches, "_fit_setting", lambda run, name, *rest: tried.append(name)
    )
    graph_launches._fitted_settings("run", (("aten::index", ((9, 4096), ())),))
    assert tried[0] == "--max-requests" and "--tp-size" not in tried and len(tried) == 4


def test_a_pool_s_size_is_followed_in_both_its_keys_and_its_values():
    from TraceLens.ModelUtils.graph_launches import _size_at

    def store(rows):
        return CapturedGraph(
            "", [_launch("lib::store_cache", ((128, 1024), (rows, 1024), (rows, 1024)))]
        )

    eager = (("lib::store_cache", ((128, 1024), (1211961, 1024), (1211961, 1024))),)
    where, predicted, traced = memory_dimension([store(1216528)], eager)
    assert (predicted, traced) == (1216528, 1211961)
    # The reservation moved both arguments; the launch is still the one followed.
    assert _size_at([store(1199000)], where, near=predicted) == 1199000


def test_only_a_switch_like_variable_is_tried_on_and_off():
    from TraceLens.ModelUtils.graph_launches import _switch_like

    assert _switch_like(None) and _switch_like("0") and _switch_like("true")
    assert not _switch_like("/opt/rocm/bin:/usr/bin") and not _switch_like(
        "unified_kv_triton"
    )


def test_an_environment_variable_holding_a_path_is_not_tried_as_a_switch(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from TraceLens.ModelUtils import engine_settings, engine_sources
    from TraceLens.ModelUtils.engine_run import EngineGraphs, EngineRun
    from TraceLens.ModelUtils.graph_launches import _setting_values

    monkeypatch.setattr(
        engine_sources,
        "engine_sources",
        lambda *args, **kwargs: SimpleNamespace(engine_checkout=tmp_path),
    )
    monkeypatch.setattr(engine_settings, "recipe_values", lambda root, model: {})
    monkeypatch.setattr(engine_settings, "compared_values", lambda roots: {})
    monkeypatch.setattr(
        engine_settings, "switch_variables", lambda roots: {"ENGINE_FAST"}
    )
    monkeypatch.setattr(engine_settings, "number_variables", lambda roots: set())
    started = EngineGraphs(
        settings_read={
            "ENGINE_JIT_DIR": "/tmp/jit",
            "ENGINE_FAST": None,
            "ENGINE_CACHE": None,
        }
    )
    run = EngineRun("sglang", "v1", "org/model", "gfx95", DeviceFacts(name="gpu"))
    assert _setting_values("ENGINE_JIT_DIR", run, started, None) == []
    # Unset: a switch only if the source reads one.
    assert _setting_values("ENGINE_FAST", run, started, None) == ["1", "0"]
    assert _setting_values("ENGINE_CACHE", run, started, None) == []


def test_running_more_of_the_engine_is_progress_only_while_a_start_fails():
    from TraceLens.ModelUtils.graph_launches import _gain

    started = (0, 19, -322, True, 1859)
    assert _gain((0, 19, -300, True, 1900)) == _gain(started)
    assert _gain((0, 20, -322, True, 1800)) > _gain(started)
    assert _gain((-7, 0, 0, False, 900)) > _gain((-7, 0, 0, False, 500))
    assert _gain(started) > _gain((-7, 0, 0, False, 1900))


def test_an_unnamed_launch_explains_the_kernels_its_op_launched_outside_graphs():
    from TraceLens.ModelUtils.engine_run import EngineGraphs
    from TraceLens.ModelUtils.graph_launches import _fit

    def graph(ops):
        launches = [
            RecordedLaunch(op, ArgumentRecord((), (), (), ()), kernels)
            for op, kernels in ops
        ]
        return CapturedGraph("", launches, context=(("capture_one", (4,)),))

    replay = frozenset({(("norm", "index_elementwise_kernel"), (4, 4))})
    eager = frozenset(
        {
            ("aten::index", "index_elementwise_kernel"),
            ("aten::gather", "vectorized_gather_kernel"),
        }
    )
    gathered = EngineGraphs(
        graphs=[graph([("lib::norm", ("norm",)), ("aten::gather", ())])]
    )
    indexed = EngineGraphs(
        graphs=[graph([("lib::norm", ("norm",)), ("aten::index", ())])]
    )
    assert (
        _fit(indexed, replay, eager)[1] == 2 and _fit(gathered, replay, eager)[1] == 1
    )


def test_a_switch_is_a_variable_the_source_reads_as_on_or_off(tmp_path):
    from TraceLens.ModelUtils.engine_settings import switch_variables

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "env.py").write_text(textwrap.dedent("""
        import os

        class Envs:
            ENGINE_DECLARED = EnvBool(False)
            ENGINE_SIZE = EnvInt(4)

        fast = os.environ.get("ENGINE_DEFAULTED", "0")
        flag = get_bool_env_var("ENGINE_HELPER")
        on = os.environ.get("ENGINE_COMPARED") == "1"
        cache = os.environ.get("ENGINE_JIT_DIR")
    """))
    assert switch_variables(tmp_path) == {
        "ENGINE_DECLARED",
        "ENGINE_DEFAULTED",
        "ENGINE_HELPER",
        "ENGINE_COMPARED",
    }


def test_one_launch_more_of_an_op_the_graph_already_launches_explains_one_kernel_more():
    from TraceLens.ModelUtils.engine_run import EngineGraphs
    from TraceLens.ModelUtils.graph_launches import _fit

    def graph(ops):
        launches = [
            RecordedLaunch(op, ArgumentRecord((), (), (), ()), ()) for op in ops
        ]
        return CapturedGraph("", launches, context=(("capture_one", (4,)),))

    replay = frozenset({(("index_kernel", "index_kernel", "mul_kernel"), (4, 4))})
    eager = frozenset({("aten::index", "index_kernel"), ("aten::mul", "mul_kernel")})
    once = EngineGraphs(graphs=[graph(["aten::index", "aten::mul", "aten::add"])])
    twice = EngineGraphs(graphs=[graph(["aten::index", "aten::index", "aten::mul"])])
    assert _fit(once, replay, eager)[1] == 2 and _fit(twice, replay, eager)[1] == 3


def test_a_replay_of_twice_a_graph_s_kernels_is_not_that_graph_s():
    from TraceLens.ModelUtils.engine_run import EngineGraphs
    from TraceLens.ModelUtils.graph_launches import _replay_pairs

    decode = CapturedGraph(
        "",
        [_launch("lib::norm", ((1, 8),), ("norm",)) for _ in range(4)],
        context=(("capture_one", (1,)),),
    )
    started = EngineGraphs(graphs=[decode])
    # A decode step of one request ran the graph; a prefill of one request ran
    # twice the kernels, the graph was not captured for it.
    pairs = dict(
        _replay_pairs(
            started, frozenset({(("norm",) * 4, (1, 1)), (("norm",) * 8, (1, 4096))})
        )
    )
    assert pairs[("norm",) * 4] == [decode] and pairs[("norm",) * 8] == []


def test_a_recipe_s_option_value_is_read_without_its_markdown_quoting():
    from TraceLens.ModelUtils.engine_settings import _stated_settings

    assert (
        _stated_settings(
            "launch with `--graph-backend-prefill breakable` and --page-size 256"
        )["--graph-backend-prefill"]
        == "breakable"
    )


def test_a_search_whose_replays_all_have_graphs_tries_one_batch_without_gain(
    monkeypatch,
):
    from types import SimpleNamespace

    from TraceLens.ModelUtils import graph_launches

    batches = []
    names = [f"SETTING_{n}" for n in range(30)]
    monkeypatch.setattr(graph_launches, "_started", lambda run: SimpleNamespace())
    monkeypatch.setattr(graph_launches, "_known_settings", lambda run: {})
    monkeypatch.setattr(
        graph_launches, "_checkpoint_settings", lambda started, known: []
    )
    monkeypatch.setattr(graph_launches, "_capture_settings", lambda *a: [])
    monkeypatch.setattr(graph_launches, "_deciding_settings", lambda *a: names)
    monkeypatch.setattr(graph_launches, "_setting_values", lambda name, *a: ["1"])
    monkeypatch.setattr(
        graph_launches, "_with_settings", lambda run, found: next(iter(found))
    )
    monkeypatch.setattr(
        graph_launches, "_start_together", lambda runs: batches.append(list(runs))
    )
    monkeypatch.setattr(graph_launches, "_described", lambda run: str(run))
    model = SimpleNamespace(
        replay_graphs=frozenset({(("k",), (1, 1))}), eager_kernels=frozenset()
    )
    # Every replay has its graph: one batch of the likeliest, then done.
    monkeypatch.setattr(
        graph_launches,
        "_fit",
        lambda started, replay, eager=frozenset(): (0, 5, -1, True, 10),
    )
    graph_launches._searched_run("run", model)
    assert len(batches) == 1
    # A replay without its graph keeps looking, a few batches.
    batches.clear()
    monkeypatch.setattr(
        graph_launches,
        "_fit",
        lambda started, replay, eager=frozenset(): (-1, 5, -1, True, 10),
    )
    graph_launches._searched_run("run", model)
    assert len(batches) == graph_launches._BATCHES_WITHOUT_GAIN
