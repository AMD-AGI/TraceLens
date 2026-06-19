
'''TraceLens perf-model "harness" runners.

Each ``run_perf_model_<op>(args)`` function constructs a synthetic event dict
matching the TraceLens trace schema, instantiates the corresponding perf-model
class, and returns ``(flops, bytes, compute_precision)``.

These functions are imported by ``validate_perf_model.py`` and registered in
its ``OP_REGISTRY``. The CSV-driven ``run_perf_model_from_event`` shares the
same dispatch surface but builds events from arbitrary
``unified_perf_summary.csv`` rows.

The module also hosts a couple of small helpers used exclusively by the
harnesses:

* :func:`_ensure_tracelens_importable` -- prepends the TraceLens repo root to
  ``sys.path`` so the ``TraceLens.PerfModel.*`` packages are importable when
  this script lives in ``TraceLens/Agent/PerfModel/validate-perf-model/``.
* :func:`_tuple_to_list` -- recursively coerce nested tuples into lists so
  the perf-model classes (which expect list-shaped event payloads) accept
  them.
* :func:`_parse_unified_attention_annotation` -- parse the vLLM execute
  annotation used by unified-attention perf models.
'''
import re
import sys
from pathlib import Path

def _ensure_tracelens_importable():
    '''Add TraceLens repo root to sys.path so perf model classes are importable.'''
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        return None


def _tuple_to_list(obj):
    '''Recursively convert tuples to lists for event dict compatibility.'''
    if isinstance(obj, tuple):
        return [_tuple_to_list(x) for x in obj]
    return obj

_TRACE_DTYPE_MAP = {
    'bf16': 'c10::BFloat16',
    'fp16': 'c10::Half',
    'fp32': 'float',
    'fp64': 'double',
    'fp8': 'c10::Float8_e4m3fnuz',
    'fp8_e4m3fn': 'c10::Float8_e4m3fn',
    'fp8_e4m3fnuz': 'c10::Float8_e4m3fnuz',
    'fp8_e5m2': 'c10::Float8_e5m2',
    'fp4': 'c10::Float4_e2m1fn_x2',
    'fp4x2': 'c10::Float4_e2m1fn_x2',
    'i8': 'signed char',
    'u8': 'unsigned char',
    'i32': 'int',
    'u16': 'unsigned short',
    'u32': 'unsigned int' }

def _trace_dtype_str(name, default):
    '''Map a user-facing dtype string ("bf16", "fp8", ...) to a trace ``Input type``.

    Returns ``default`` when ``name`` is None or empty (preserves the original
    hard-coded behavior of each ``run_perf_model_*`` runner). When ``name`` is
    already a fully-qualified trace string (e.g. ``"c10::BFloat16"``), it is
    returned unchanged so callers can pass either form.
    '''
    if name is None or name == '':
        return default
    key = str(name).strip()
    lower = key.lower()
    if lower in _TRACE_DTYPE_MAP:
        return _TRACE_DTYPE_MAP[lower]
    if '::' in key or key in (
        'unsigned short', 'unsigned char', 'unsigned int', 'signed char',
        'int', 'float', 'double', 'Half', 'BFloat16',
    ):
        return key
    raise ValueError(
        f'Unknown dtype string {name}. Known short names: {sorted(_TRACE_DTYPE_MAP)}'
    )


def _moe_dtypes_from_args(args, defaults):
    """Resolve ``(in_dtype, w_dtype, out_dtype)`` trace strings from ``args``.

    ``defaults`` is a 3-tuple of trace strings used when the corresponding CLI
    flag is unset. This keeps backward-compatible behavior for OP_REGISTRY
    entries that don't pass dtype overrides.
    """
    in_t = _trace_dtype_str(getattr(args, 'in_dtype', None), defaults[0])
    w_t = _trace_dtype_str(getattr(args, 'w_dtype', None), defaults[1])
    out_t = _trace_dtype_str(getattr(args, 'out_dtype', None), defaults[2])
    return (in_t, w_t, out_t)


def _dtype_from_args(args, attr_name, default_trace_str):
    '''Resolve a single dtype attribute on ``args`` to a trace ``Input type``.

    Convenience wrapper that mirrors :func:`_moe_dtypes_from_args` for
    non-MoE runners that only need one or two dtype kwargs.
    '''
    return _trace_dtype_str(getattr(args, attr_name, None), default_trace_str)


def _kv_dtype_from_args(args, default_trace_str):
    '''Resolve ``args.kv_dtype`` (paged-attention KV cache dtype).'''
    return _trace_dtype_str(getattr(args, 'kv_dtype', None), default_trace_str)


def run_perf_model_gemm(args):
    '''Perf model for ``aiter::gemm_a8w8_blockscale_ck`` (FP8 / INT8 block-scaled GEMM).

    Dtype kwargs read from ``args`` (defaults match
    :func:`tests.gemm.test_gemm_a8w8_blockscale`):

    * ``in_dtype`` -- activation storage (``"i8"``, ``"u8"``, ``"fp8"`` -- all 1 BPE).
    * ``w_dtype``  -- weight storage (``"i8"``, ``"u8"``, ``"fp8"`` -- all 1 BPE).
    * ``scale_dtype`` -- block-scale dtype (``"fp32"``).
    * ``block_k`` -- per-K-block tile size for scales (default 128).

    ``out_dtype`` is accepted for symmetry but the perf-model class hardcodes
    BF16 output (see ``perf_model_extensions.py:gemm_a8w8_blockscale``); it is
    forwarded to the test runner only.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import gemm_a8w8_blockscale as cls
    K = args.K
    N = args.N
    M = args.M
    if not getattr(args, 'block_k', None):
        getattr(args, 'block_k', None)
    block_k = 128
    nkb = (K + block_k - 1) // block_k
    in_t = _dtype_from_args(args, 'in_dtype', 'unsigned char')
    w_t = _dtype_from_args(args, 'w_dtype', 'unsigned char')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'aiter::gemm_a8w8_blockscale_ck',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K],
                [
                    M,
                    nkb],
                [
                    N,
                    nkb]],
            'Input type': [
                in_t,
                w_t,
                scl_t,
                scl_t],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K,
                    1],
                [
                    nkb,
                    1],
                [
                    nkb,
                    1]] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_fmoe(args):
    '''Perf model for ``aiter::fmoe_fp8_blockscale_g1u1``.

    Dtype kwargs read from ``args`` (all optional; defaults match
    :func:`tests.moe.test_fmoe_fp8_blockscale_g1u1`):

    * ``in_dtype``    -- activation dtype. ``"bf16"`` exercises the in-kernel
      quant path; ``"fp8"`` does caller-side ``pertoken_quant``. Default ``"bf16"``.
    * ``w_dtype``     -- expert weight dtype. Kernel only supports ``"fp8"``.
    * ``out_dtype``   -- output buffer dtype. Kernel only supports ``"bf16"``.
    * ``block_n`` / ``block_k`` -- block-scale tile sizes (default ``128``).
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import moe_aiter_fused_blockscale
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(args, defaults = ('c10::BFloat16', 'c10::Float8_e4m3fnuz', 'c10::BFloat16'))
    if not getattr(args, 'block_n', None):
        getattr(args, 'block_n', None)
    block_n = 128
    if not getattr(args, 'block_k', None):
        getattr(args, 'block_k', None)
    block_k = 128
    concrete = [
        None] * 15
    concrete[8] = topk
    concrete[13] = block_n
    concrete[14] = block_k
    event = {
        'name': 'aiter::fmoe_fp8_blockscale_g1u1',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    M,
                    K],
                [
                    E,
                    N * 2,
                    K],
                [
                    E,
                    K,
                    N]],
            'Input type': [
                out_t,
                in_t,
                w_t,
                w_t],
            'Concrete Inputs': concrete } }
    model = moe_aiter_fused_blockscale(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def _run_perf_model_activation(args, fn_name, cls_name):
    '''Generic perf-model runner for ``silu_and_mul`` / ``gelu_and_mul`` / ``gelu_tanh_and_mul``.

    Reads ``args.in_dtype`` (slot ``[1]``, the gate||up input) and
    ``args.out_dtype`` (slot ``[0]``). Defaults to BF16 for both.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions import perf_model_extensions as ext
    cls = getattr(ext, cls_name)
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    out_t = _dtype_from_args(args, 'out_dtype', 'c10::BFloat16')
    event = {
        'name': f'''aiter::{fn_name}''',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    M,
                    2 * N]],
            'Input type': [
                out_t,
                in_t],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    2 * N,
                    1]] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def _make_activation_model_fn(fn_name, cls_name):
    def runner(args):
        return _run_perf_model_activation(args, fn_name, cls_name)
    return runner


def run_perf_model_rms_norm(args):
    '''Perf model for ``aiter::rms_norm`` (CK).

    Reads ``in_dtype`` (slot ``[0]``), ``w_dtype`` (slot ``[1]``). Output
    traffic is computed using ``in_dtype`` (kernel out matches input).
    Defaults match :func:`tests.rmsnorm.test_rms_norm`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import aiter_rms_norm
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    event = {
        'name': 'aiter::rms_norm',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    N],
                [],
                []],
            'Input type': [
                in_t,
                w_t,
                'Scalar',
                'Scalar'],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    1],
                [],
                []] } }
    model = aiter_rms_norm(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_rmsnorm(args):
    '''Perf model for ``aiter::rmsnorm`` (out-first API).

    Note that the perf model class reads slot ``[1]`` for the input dtype
    (the API is ``rmsnorm(out, inp, weight, eps)`` so slot ``[0]`` is the
    out buffer). Both slots therefore receive ``in_dtype``; ``w_dtype`` goes
    to slot ``[2]``.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import aiter_rmsnorm
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    out_t = _dtype_from_args(args, 'out_dtype', in_t)
    event = {
        'name': 'aiter::rmsnorm',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    N],
                []],
            'Input type': [
                out_t,
                in_t,
                w_t,
                'Scalar'],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    1],
                []] } }
    model = aiter_rmsnorm(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_add_rmsnorm(args):
    '''Perf model for ``aiter::add_rmsnorm`` (fused residual-add + RMSNorm).

    Slot ``[1]`` is the input (read by the perf-model class for byte-counting
    and compute precision). All BF16 traffic uses ``in_dtype``.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import aiter_rmsnorm2d_fwd_with_add_ck
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    event = {
        'name': 'aiter::add_rmsnorm',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    N],
                [],
                []],
            'Input type': [
                in_t,
                in_t,
                in_t,
                in_t,
                w_t,
                'Scalar',
                'Scalar'],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    1],
                [],
                []] } }
    model = aiter_rmsnorm2d_fwd_with_add_ck(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_rmsnorm_dynamicquant(args):
    '''Perf model for ``aiter::rmsnorm2d_fwd_with_dynamicquant_ck``.

    Reads slot ``[1]`` for the input dtype. FP8 output bytes (slot ``[0]``)
    and per-token FP32 scale bytes (slot ``[2]``) are hardcoded by the
    perf-model class, so changing ``out_dtype`` only affects the test runner
    side.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import aiter_rmsnorm2d_fwd_with_dynamicquant_ck
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    out_t = _dtype_from_args(args, 'out_dtype', 'unsigned char')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'aiter::rmsnorm2d_fwd_with_dynamicquant_ck',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    M,
                    1],
                [
                    N],
                [],
                []],
            'Input type': [
                out_t,
                in_t,
                scl_t,
                w_t,
                'Scalar',
                'Scalar'],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    1,
                    1],
                [
                    1],
                [],
                []] } }
    model = aiter_rmsnorm2d_fwd_with_dynamicquant_ck(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dynamic_quant(args):
    '''Perf model for ``aiter::dynamic_per_token_scaled_quant``.

    Reads slot ``[0]`` for output dtype, slot ``[1]`` for input, slot ``[2]``
    for scales. Compute precision is taken from input.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import per_group_quant
    N = args.N
    M = args.M
    out_t = _dtype_from_args(args, 'out_dtype', 'c10::Float8_e4m3fnuz')
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'aiter::dynamic_per_token_scaled_quant',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    M,
                    N],
                [
                    M,
                    1]],
            'Input type': [
                out_t,
                in_t,
                scl_t],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    N,
                    1],
                [
                    1,
                    1]] } }
    model = per_group_quant(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_gemm_a16w16(args):
    '''Perf model for ``aiter::gemm_a16w16_atomic_`` (BF16 / FP16 GEMM).

    Dtype kwargs (``in_dtype`` / ``w_dtype`` / ``out_dtype``); kernel requires
    all three to match. Defaults to ``"bf16"``.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import gemm_a16w16_atomic_
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    out_t = _dtype_from_args(args, 'out_dtype', 'c10::BFloat16')
    event = {
        'name': 'aiter::gemm_a16w16_atomic_',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K],
                [],
                [
                    M,
                    N]],
            'Input type': [
                in_t,
                w_t,
                '',
                out_t] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = gemm_a16w16_atomic_(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_moe_unfused_up(args):
    '''Perf model for ``aiter::moe_cktile2stages_gemm1_ck`` (CK-Tile MoE up).

    Dtype kwargs (defaults reflect the BF16 / FP4-weight a16w4 path used by
    the unparameterized harness):

    * ``in_dtype``   -- activation dtype (default ``"bf16"``).
    * ``w_dtype``    -- packed FP4 weight dtype (default ``"fp4x2"``).
    * ``out_dtype``  -- output dtype (default ``"bf16"``).
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import moe_aiter_unfused_up
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(args, defaults = ('c10::BFloat16', 'c10::Float4_e2m1fn_x2', 'c10::BFloat16'))
    event = {
        'name': 'aiter::moe_cktile2stages_gemm1_ck',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    E,
                    N * 2,
                    K],
                [
                    M,
                    topk,
                    N]],
            'Input type': [
                in_t,
                w_t,
                out_t] } }
    model = moe_aiter_unfused_up(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_moe_unfused_down(args):
    '''Perf model for ``aiter::moe_cktile2stages_gemm2_ck`` (CK-Tile MoE down).

    Same dtype kwargs as :func:`run_perf_model_moe_unfused_up`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import moe_aiter_unfused_down
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(args, defaults = ('c10::BFloat16', 'c10::Float4_e2m1fn_x2', 'c10::BFloat16'))
    event = {
        'name': 'aiter::moe_cktile2stages_gemm2_ck',
        'args': {
            'Input Dims': [
                [
                    M,
                    topk,
                    N],
                [
                    E,
                    K,
                    N],
                [
                    M,
                    K]],
            'Input type': [
                in_t,
                w_t,
                out_t] } }
    model = moe_aiter_unfused_down(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_ck_moe_stage1(args):
    '''Perf model for ``aiter::ck_moe_stage1`` (CK MoE up + fused activation).

    Dtype kwargs:

    * ``in_dtype``   -- activation dtype. ``"bf16"``/``"fp16"`` (No quant) or
      ``"fp8"``/``"i8"``/``"fp4x2"`` (quantized A). Default ``"bf16"``.
    * ``w_dtype``    -- weight dtype (``"bf16"``/``"fp16"``/``"fp8"``/``"i8"``/
      ``"fp4x2"``). Default ``"bf16"``.
    * ``out_dtype``  -- output dtype (``"bf16"`` / ``"fp16"``). Default
      ``"bf16"``.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import moe_aiter_ck_stage1
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(args, defaults = ('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16'))
    event = {
        'name': 'aiter::ck_moe_stage1',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    E,
                    N * 2,
                    K],
                [
                    E,
                    K,
                    N],
                [
                    M * topk],
                [
                    E],
                [
                    E],
                [
                    M,
                    topk,
                    N]],
            'Input type': [
                in_t,
                w_t,
                out_t] } }
    model = moe_aiter_ck_stage1(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_ck_moe_stage2(args):
    '''Perf model for ``aiter::ck_moe_stage2`` (CK MoE down + topk reduce).

    Same dtype kwargs as :func:`run_perf_model_ck_moe_stage1`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import moe_aiter_ck_stage2
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(args, defaults = ('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16'))
    event = {
        'name': 'aiter::ck_moe_stage2',
        'args': {
            'Input Dims': [
                [
                    M,
                    topk,
                    N],
                [
                    E,
                    N * 2,
                    K],
                [
                    E,
                    K,
                    N],
                [
                    M * topk],
                [
                    E],
                [
                    E],
                [
                    M,
                    K]],
            'Input type': [
                in_t,
                w_t,
                out_t] } }
    model = moe_aiter_ck_stage2(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_sglang_fused_moe_invoke(args):
    '''Perf model for ``sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel``.

    A single SGLang Triton ``invoke_fused_moe_kernel`` launch (one grouped GEMM).
    This same op name covers both the gate/up and the down projection; the
    perf-model class derives ``(M_work, N, K)`` generically from the args, so a
    single representative shape validates either direction.

    Args read:

    * ``M``     -- number of input tokens (pre-expansion).
    * ``N``     -- weight output dim (for gate/up pass this is ``2*inter_dim``).
    * ``K``     -- weight contraction dim (hidden_dim for gate/up).
    * ``E``     -- number of experts.
    * ``topk``  -- experts per token.
    * ``in_dtype``  -- activation dtype (default ``"bf16"``; quantized to fp8 in-kernel).
    * ``w_dtype``   -- expert weight dtype (default ``"fp8"``).
    * ``out_dtype`` -- output dtype (default ``"bf16"``).

    The event layout mirrors the trace ``Input Dims``:
        [0] A            = (M, K)
        [1] B            = (E, N, K)
        [2] C            = ()                (output, dims unrecorded)
        [3] C buffer     = (M*topk, N)
        [4] ()
        [5] B_scale      = (E,)
        [6] ()
        [7] topk_weights = (M, topk)
        [8] topk_ids     = (M, topk)
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
        moe_triton_invoke_grouped_gemm,
    )
    (M, K, N, E, topk) = (args.M, args.K, args.N, args.E, args.topk)
    (in_t, w_t, out_t) = _moe_dtypes_from_args(
        args, defaults=('c10::BFloat16', 'c10::Float8_e4m3fnuz', 'c10::BFloat16'))
    event = {
        'name': 'sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel',
        'args': {
            'Input Dims': [
                [M, K],
                [E, N, K],
                [],
                [M * topk, N],
                [],
                [E],
                [],
                [M, topk],
                [M, topk],
                [M * topk + E * 64],
                [E * 2],
                [1],
                []],
            'Input type': [
                in_t,
                w_t,
                '',
                out_t,
                '',
                'float',
                '',
                'float',
                'int',
                'int',
                'int',
                'int',
                ''] } }
    model = moe_triton_invoke_grouped_gemm(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_flash_attn(args):
    '''Perf model for ``aiter::_flash_attn_forward`` (CK flash attention).

    Reads ``in_dtype`` for Q/K/V slots (kernel requires matching dtypes).

    Note
    ----
    The upstream perf-model class (``aiter__flash_attn_forward``) hardcodes
    ``bytes_per_element=2`` in its ``bytes()`` method, so changing
    ``in_dtype`` does not affect predicted bytes; it only flows through to
    the test runner. ``get_compute_precision()`` returns ``None`` for this
    class (it never populates ``param_details["dtype_A_B"]``).
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.perf_model import aiter__flash_attn_forward
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    concrete = [
        ''] * 12
    concrete[3] = '0.0'
    concrete[5] = 'true'
    event = {
        'name': 'aiter::_flash_attn_forward',
        'args': {
            'Input Dims': [
                [
                    1,
                    S,
                    H_Q,
                    d],
                [
                    1,
                    S,
                    H_KV,
                    d],
                [
                    1,
                    S,
                    H_KV,
                    d]],
            'Input type': [
                in_t,
                in_t,
                in_t],
            'Concrete Inputs': concrete } }
    model = aiter__flash_attn_forward(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_fmha_v3(args):
    '''Perf model for ``aiter::wrapper_fmha_v3_fwd`` (FMHA v3 dense forward).

    Same dtype constraints + same hardcoded-bytes caveat as
    :func:`run_perf_model_flash_attn`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.perf_model import aiter__fmha_v3_forward
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    concrete = [
        ''] * 9
    concrete[4] = '0.0'
    concrete[6] = 'true'
    event = {
        'name': 'aiter::wrapper_fmha_v3_fwd',
        'args': {
            'Input Dims': [
                [],
                [
                    1,
                    S,
                    H_Q,
                    d],
                [
                    1,
                    S,
                    H_KV,
                    d],
                [
                    1,
                    S,
                    H_KV,
                    d]],
            'Input type': [
                '',
                in_t,
                in_t,
                in_t],
            'Concrete Inputs': concrete } }
    model = aiter__fmha_v3_forward(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_mha_varlen(args):
    '''Perf model for ``aiter::mha_varlen_fwd`` (CK varlen flash attention).

    Reads slot ``[0]`` for Q dtype. Bytes are computed as
    ``name2bpe(dtype_Q) or 2``, so passing ``--in_dtype fp16`` yields BF16
    -> FP16 byte counts (both 2 BPE) and equivalent compute precision.

    Annotation format
    -----------------
    Uses the canonical ``execute_*`` format shared by all attention harnesses::

        execute_0_context_N(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                            _generation_0(sq0sk0sqsq0sqsk0)

    where:

    * ``c_sq``  = total Q tokens across all context sequences
    * ``c_sk``  = total KV tokens across all context sequences (= ``total_kv``)
    * ``c_sqsq``= Σ sq_i²  (causal correction)
    * ``c_sqsk``= Σ sq_i·sk_i  (= c_sqsq for self-attention where sq_i = sk_i)

    The generation block is always zero because this is a prefill-only kernel.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import mha_varlen_fwd
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    c_sq = getattr(args, '_varlen_c_sq', S)
    c_sqsq = getattr(args, '_varlen_c_sqsq', S * S)
    total_q = getattr(args, '_varlen_total_q', S)
    total_kv = getattr(args, '_varlen_total_kv', S)
    c_sk = total_kv
    c_sqsk = getattr(args, '_varlen_c_sqsk', c_sqsq)
    n_ctx = getattr(args, '_varlen_n_ctx', 1)
    annotation = f'''execute_0_context_{n_ctx}(sq{c_sq}sk{c_sk}sqsq{c_sqsq}sqsk{c_sqsk})_generation_0(sq0sk0sqsq0sqsk0)'''
    event = {
        'name': 'aiter::mha_varlen_fwd',
        'annotation': annotation,
        'args': {
            'Input Dims': [
                [
                    total_q,
                    H_Q,
                    d],
                [
                    total_kv,
                    H_KV,
                    d]],
            'Input type': [
                in_t,
                in_t] } }
    model = mha_varlen_fwd(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_fmha_v3_varlen(args):
    '''Perf model for ``aiter::fmha_v3_varlen_fwd`` (FMHA v3 varlen forward).

    Same dtype + annotation semantics as :func:`run_perf_model_mha_varlen`:
    uses the canonical ``execute_*`` format with a zero-generation block.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import aiter_fmha_v3_varlen_fwd
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    c_sq = getattr(args, '_varlen_c_sq', S)
    c_sqsq = getattr(args, '_varlen_c_sqsq', S * S)
    total_q = getattr(args, '_varlen_total_q', S)
    total_kv = getattr(args, '_varlen_total_kv', S)
    c_sk = total_kv
    c_sqsk = getattr(args, '_varlen_c_sqsk', c_sqsq)
    n_ctx = getattr(args, '_varlen_n_ctx', 1)
    annotation = f'''execute_0_context_{n_ctx}(sq{c_sq}sk{c_sk}sqsq{c_sqsq}sqsk{c_sqsk})_generation_0(sq0sk0sqsq0sqsk0)'''
    event = {
        'name': 'aiter::fmha_v3_varlen_fwd',
        'annotation': annotation,
        'args': {
            'Input Dims': [
                [
                    total_q,
                    H_Q,
                    d],
                [
                    total_kv,
                    H_KV,
                    d],
                [
                    total_kv,
                    H_KV,
                    d]],
            'Input type': [
                in_t,
                in_t,
                in_t] } }
    model = aiter_fmha_v3_varlen_fwd(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_unquantized_gemm(args):
    '''Perf model for ``vllm::rocm_unquantized_gemm`` (BF16 / FP16 GEMM).

    Dtype kwargs (defaults match :func:`tests.gemm.test_vllm_unquantized_gemm`):

    * ``in_dtype`` / ``w_dtype`` -- activation / weight (default ``"bf16"``).
    * ``bias_dtype`` -- bias dtype if a bias is supplied (default unset =
      no bias). Set this to enable the bias-traffic bytes term.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import vllm_rocm_unquantized_gemm as cls
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    bias_t = _dtype_from_args(args, 'bias_dtype', '')
    event = {
        'name': 'vllm::rocm_unquantized_gemm',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K]],
            'Input type': [
                in_t,
                w_t,
                bias_t],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K,
                    1]] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_triton_group_quant_fp8(args):
    '''Perf model for ``vllm::triton_per_token_group_quant_fp8``.

    Reads slot ``[0]`` for input dtype. FP8 output (1 BPE) and FP32 scales
    (4 BPE) are hardcoded by the upstream class.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import vllm_triton_per_token_group_quant_fp8 as cls
    N = args.N
    M = args.M
    if not getattr(args, 'group_size', 128):
        getattr(args, 'group_size', 128)
    group_size = 128
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    event = {
        'name': 'vllm::triton_per_token_group_quant_fp8',
        'args': {
            'Input Dims': [
                (M, N),
                ()],
            'Input type': [
                in_t,
                'Scalar'],
            'Concrete Inputs': [
                '',
                str(group_size)] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_rmsnorm_fp8_group_quant(args):
    '''Perf model for ``vllm::rocm_aiter_rmsnorm_fp8_group_quant``.

    Reads slot ``[0]`` for x dtype (input). The FP8 quant output and FP32
    scales are hardcoded by the upstream class.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import vllm_rocm_aiter_rmsnorm_fp8_group_quant as cls
    N = args.N
    M = args.M
    if not getattr(args, 'group_size', 128):
        getattr(args, 'group_size', 128)
    group_size = 128
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    event = {
        'name': 'vllm::rocm_aiter_rmsnorm_fp8_group_quant',
        'args': {
            'Input Dims': [
                (M, N),
                (N,),
                (),
                ()],
            'Input type': [
                in_t,
                w_t,
                'Scalar',
                'Scalar'],
            'Input Strides': [
                (N, 1),
                (1,),
                (),
                ()],
            'Concrete Inputs': [
                '',
                '',
                '',
                str(group_size)] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_rmsnorm_add_fp8_group_quant(args):
    '''Perf model for ``vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant``.'''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant as cls
    N = args.N
    M = args.M
    if not getattr(args, 'group_size', 128):
        getattr(args, 'group_size', 128)
    group_size = 128
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::BFloat16')
    event = {
        'name': 'vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant',
        'args': {
            'Input Dims': [
                (M, N),
                (M, N),
                (N,),
                (),
                ()],
            'Input type': [
                in_t,
                in_t,
                w_t,
                'Scalar',
                'Scalar'],
            'Input Strides': [
                (N, 1),
                (N, 1),
                (1,),
                (),
                ()],
            'Concrete Inputs': [
                '',
                '',
                '',
                '',
                str(group_size)] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_triton_gemm_a8w8_blockscale(args):
    '''Perf model for the vLLM Triton wrapper around ``gemm_a8w8_blockscale``.

    Same dtype semantics as :func:`run_perf_model_gemm`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import gemm_a8w8_blockscale as cls
    K = args.K
    N = args.N
    M = args.M
    if not getattr(args, 'block_k', None):
        getattr(args, 'block_k', None)
    block_k = 128
    nkb = (K + block_k - 1) // block_k
    in_t = _dtype_from_args(args, 'in_dtype', 'unsigned char')
    w_t = _dtype_from_args(args, 'w_dtype', 'unsigned char')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'vllm::rocm_aiter_triton_gemm_a8w8_blockscale',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K],
                [
                    M,
                    nkb],
                [
                    N,
                    nkb]],
            'Input type': [
                in_t,
                w_t,
                scl_t,
                scl_t],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K,
                    1],
                [
                    nkb,
                    1],
                [
                    nkb,
                    1]] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_unified_attention(args):
    """Perf model for ``vllm::unified_attention_with_output`` (paged decode).

    Reads slot ``[0]`` for Q dtype (via ``InferenceAttention.get_param_details``).
    The KV-cache dtype is conveyed via slot ``[1]`` for documentation, even
    though the upstream class does not consume it -- bytes are
    ``name2bpe(dtype_Q) or 2``.

    Annotation flow
    ---------------
    All scenarios converge on a canonical vLLM-format annotation:

        execute_<iter>_context_<c_n>(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                      _generation_<g_n>(sq<g_sq>sk<g_sk>sqsq<g_sqsq>sqsk<g_sqsk>)

    ``vllm_unified_attention_with_output`` is a paged-decode-only kernel; any
    prefill in the same scheduler step runs through a separate flash-attention
    kernel.  Therefore c_* are always 0 here and all decode statistics live in
    the g_* fields:

        g_sq   = n_decode              (one Q token per decode sequence)
        g_sk   = n_decode * ctx_len    (total KV tokens accessed by the kernel)
        g_sqsq = g_sq  (same; no causal masking across sequences)
        g_sqsk = g_sk  (same)

    ``sum_q`` and ``sum_kv`` are then derived by parsing this annotation through
    ``_parse_unified_attention_annotation``, matching the identical path used for
    real vLLM traces.  This ensures the perf model's FLOPs and bytes calculations
    are consistent regardless of the annotation source.
    """
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import vllm_unified_attention_with_output as cls
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    kv_t = _kv_dtype_from_args(args, 'c10::Float8_e4m3fnuz')
    annotation_str = getattr(args, 'annotation', None)
    if not _parse_unified_attention_annotation(annotation_str):
        num_decode_seqs = getattr(args, 'num_decode_seqs', None)
        ctx_len_arg = getattr(args, 'ctx_len', None)
        p_len = int(getattr(args, 'prefill_seq_len', 0) or 0)
        _seq_fallback = int(getattr(args, 'seq_len', None) or 256)
        n_decode = int(num_decode_seqs) if num_decode_seqs is not None else _seq_fallback
        kv = int(ctx_len_arg) if ctx_len_arg is not None else _seq_fallback
        g_sqsk = n_decode * kv
        if p_len > 0:
            c_sqsk = p_len * p_len
            annotation_str = f'execute_0_context_1(sq{p_len}sk{p_len}sqsq{c_sqsk}sqsk{c_sqsk})_generation_{n_decode}(sq{n_decode}sk{g_sqsk}sqsq{n_decode}sqsk{g_sqsk})'
        else:
            annotation_str = f'execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_{n_decode}(sq{n_decode}sk{g_sqsk}sqsq{n_decode}sqsk{g_sqsk})'
    parsed = _parse_unified_attention_annotation(annotation_str)
    if parsed is None:
        raise ValueError(f'Cannot parse vllm_unified_attention annotation: {annotation_str!r}')
    sum_q = parsed['ctx_sq'] + parsed['gen_sq']
    sum_kv = parsed['ctx_sk'] + parsed['gen_sk']
    event = {
        'name': 'vllm::unified_attention_with_output',
        'annotation': annotation_str,
        'args': {
            'Input Dims': [[sum_q, H_Q, d], [sum_kv, H_KV, d]],
            'Input type': [in_t, kv_t]
        }
    }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def _parse_unified_attention_annotation(annotation):
    '''Parse vLLM execute_..._context_..._generation_(...) annotation.

    Returns a dict with ctx_req/ctx_sq/ctx_sk/ctx_sqsq/ctx_sqsk and
    gen_req/gen_sq/gen_sk/gen_sqsq/gen_sqsk, or None if the annotation is
    falsy / does not match.
    '''
    if not annotation:
        return None
    pat = re.compile('execute_(?P<iter>\\d+)_context_(?P<ctx_req>\\d+)\\(sq(?P<ctx_sq>\\d+)sk(?P<ctx_sk>\\d+)sqsq(?P<ctx_sqsq>\\d+)sqsk(?P<ctx_sqsk>\\d+)\\)_generation_(?P<gen_req>\\d+)\\(sq(?P<gen_sq>\\d+)sk(?P<gen_sk>\\d+)sqsq(?P<gen_sqsq>\\d+)sqsk(?P<gen_sqsk>\\d+)\\)')
    m = pat.search(str(annotation))
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def run_perf_model_unified_attention(args):
    '''Perf model for aiter triton ``unified_attention`` (paged KV / varlen).

    Reuses ``vllm_unified_attention_with_output`` (same kernel under the hood).

    Annotation flow
    ---------------
    Identical to :func:`run_perf_model_vllm_unified_attention`: all scenarios
    converge on the canonical ``execute_*`` format::

        execute_<iter>_context_<c_n>(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                      _generation_<g_n>(sq<g_sq>sk<g_sk>sqsq<g_sqsq>sqsk<g_sqsk>)

    If ``args.annotation`` already carries a valid ``execute_*`` string it is
    passed through verbatim.  Otherwise a synthetic annotation is built from
    ``args.num_decode_seqs`` / ``args.ctx_len`` / ``args.prefill_seq_len``
    (falling back to ``args.seq_len`` as a single-sequence decode scenario).

    Dtype kwargs ``in_dtype`` / ``kv_dtype`` mirror
    :func:`run_perf_model_vllm_unified_attention`.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import vllm_unified_attention_with_output as cls
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    kv_t = _kv_dtype_from_args(args, 'c10::Float8_e4m3fnuz')
    annotation_str = getattr(args, 'annotation', None)
    if not _parse_unified_attention_annotation(annotation_str):
        num_decode_seqs = getattr(args, 'num_decode_seqs', None)
        ctx_len_arg = getattr(args, 'ctx_len', None)
        p_len = int(getattr(args, 'prefill_seq_len', 0) or 0)
        _seq_fallback = int(getattr(args, 'seq_len', None) or 256)
        n_decode = int(num_decode_seqs) if num_decode_seqs is not None else _seq_fallback
        kv = int(ctx_len_arg) if ctx_len_arg is not None else _seq_fallback
        g_sqsk = n_decode * kv
        if p_len > 0:
            c_sqsk = p_len * p_len
            annotation_str = f'execute_0_context_1(sq{p_len}sk{p_len}sqsq{c_sqsk}sqsk{c_sqsk})_generation_{n_decode}(sq{n_decode}sk{g_sqsk}sqsq{n_decode}sqsk{g_sqsk})'
        else:
            annotation_str = f'execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_{n_decode}(sq{n_decode}sk{g_sqsk}sqsq{n_decode}sqsk{g_sqsk})'
    parsed = _parse_unified_attention_annotation(annotation_str)
    if parsed is None:
        raise ValueError(f'Cannot parse unified_attention annotation: {annotation_str!r}')
    sum_q = parsed['ctx_sq'] + parsed['gen_sq']
    sum_kv = parsed['ctx_sk'] + parsed['gen_sk']
    event = {
        'name': 'aiter::unified_attention',
        'annotation': annotation_str,
        'args': {
            'Input Dims': [[sum_q, H_Q, d], [sum_kv, H_KV, d]],
            'Input type': [in_t, kv_t]
        }
    }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_gemm_with_dynamic_quant(args):
    '''Perf model for ``vllm::gemm_with_dynamic_quant`` (Quark OCP MX FP4 GEMM).

    Dtype kwargs:

    * ``in_dtype`` -- activation dtype (``"bf16"`` / ``"fp16"``). The perf
      model uses this for both the activation traffic AND the output traffic
      (the class only reads slot ``[0]``).
    * ``w_dtype`` -- weight dtype (``"fp4x2"``, default). Forwarded to the
      test runner; perf-model bytes for the weight is computed implicitly
      from ``in_dtype`` (limitation of the upstream class).
    * ``scale_dtype`` -- per-32 OCP MX scale dtype (default ``"u8"``).
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.perf_model import vllm_gemm_with_dynamic_quant as cls
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'unsigned char')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'unsigned char')
    event = {
        'name': 'vllm::gemm_with_dynamic_quant',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K // 2],
                [
                    N,
                    K // 32]],
            'Input type': [
                in_t,
                w_t,
                scl_t],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K // 2,
                    1],
                [
                    K // 32,
                    1]] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_vllm_gdn_attention_core(args):
    '''Perf model for ``vllm::gdn_attention_core`` (Gated Delta Network).

    Reads slot ``[0]`` for compute precision (mixed_qkv dtype). The ``bytes()``
    method hardcodes 2 BPE so changing dtype only affects the test runner
    side.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import gdn_attention_core as cls
    T = args.seq_len
    H_V = args.num_heads_kv
    d_k = args.head_dim
    d_v = args.head_dim
    H_K = H_V // 2
    D = 2 * H_K * d_k + H_V * d_v
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    annotation = f'''attn_0_0_{T}_{T * T}_0_0_0'''
    event = {
        'name': 'vllm::gdn_attention_core',
        'annotation': annotation,
        'args': {
            'Input Dims': [
                [
                    T,
                    D],
                [
                    T,
                    H_V],
                [
                    T,
                    H_V],
                [
                    T,
                    H_V,
                    d_v],
                []],
            'Input type': [
                in_t,
                in_t,
                in_t,
                in_t,
                ''] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_atom_flydsl_preshuffle_gemm_a8(args):
    '''Perf model for ``aiter::gemm_a8w8_bpreshuffle`` (FlyDSL A8W8 GEMM).

    Maps to ``gemm_a8w8_blockscale``: FLOPs ``2*M*N*K``, bytes A/B = 1 byte
    (FP8), output = 2 bytes (BF16). The bpreshuffle layout is layout-only, so
    flops/bytes match the block-scale model.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import gemm_a8w8_blockscale as cls
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::Float8_e4m3fn')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::Float8_e4m3fn')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'aiter::gemm_a8w8_bpreshuffle',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K],
                [
                    M,
                    1],
                [
                    N,
                    1]],
            'Input type': [
                in_t,
                w_t,
                scl_t,
                scl_t],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K,
                    1],
                [
                    1,
                    1],
                [
                    1,
                    1]] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_atom_flydsl_gdr_decode(args):
    '''Perf model for ``aiter::linear_attention_with_output_base`` (FlyDSL GDR).

    Maps to ``gdn_attention_core`` -- the op signature
    ``(mixed_qkv, b, a, core_attn_out, layer_name)`` matches the layout the
    model parses. ``--seq_len`` is the GDN total-token count (decode: one token
    per sequence), injected via the GDN ``c_sq`` annotation field.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import gdn_attention_core as cls
    T = args.seq_len
    H_V = args.num_heads_kv
    d_k = args.head_dim
    d_v = args.head_dim
    H_K = max(1, H_V // 2)
    D = 2 * H_K * d_k + H_V * d_v
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    annotation = f'''attn_0_0_{T}_0_0_0_0'''
    event = {
        'name': 'aiter::linear_attention_with_output_base',
        'annotation': annotation,
        'args': {
            'Input Dims': [
                [
                    T,
                    D],
                [
                    T,
                    H_V],
                [
                    T,
                    H_V],
                [
                    T,
                    H_V,
                    d_v],
                []],
            'Input type': [
                in_t,
                in_t,
                in_t,
                in_t,
                ''] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())

import importlib.util as _ilu
import os as _os
_DSV3_EXT_DEFAULT = '/home/devashah/dsv3_analysis_output/decode_only/unified_perf_summary_triage_extension.py'
_DSV3_EXT_CACHE = { }

def _load_dsv3_extension():
    '''Load (and cache) the DSV3 triage extension module.'''
    path = _os.environ.get('DSV3_TRIAGE_EXTENSION', _DSV3_EXT_DEFAULT)
    cached = _DSV3_EXT_CACHE.get(path)
    if cached is not None:
        return cached
    _ensure_tracelens_importable()
    spec = _ilu.spec_from_file_location('dsv3_triage_extension', path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load DSV3 triage extension from '{path}'. "
            "Set DSV3_TRIAGE_EXTENSION or regenerate via run_other_bucket_triage.py."
        )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DSV3_EXT_CACHE[path] = mod
    return mod


def run_perf_model_dsv3_flydsl_hgemm(args):
    '''Perf model for ``sglang_profiler::gemm_kernels_flydsl_hgemm_54`` (BF16 GEMM).'''
    mod = _load_dsv3_extension()
    cls = mod.sglang_flydsl_hgemm
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    event = {
        'name': 'sglang_profiler::gemm_kernels_flydsl_hgemm_54',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    N,
                    K],
                [],
                []],
            'Input type': [
                in_t,
                in_t,
                '',
                ''],
            'Input Strides': [
                [
                    K,
                    1],
                [
                    K,
                    1],
                [],
                []] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_batched_gemm_a8w8(args):
    '''Perf model for sglang_profiler::batched_gemm_a8w8 (batched A8W8 GEMM).'''
    mod = _load_dsv3_extension()
    cls = mod.sglang_batched_gemm_a8w8
    (M, N, K, E) = (args.M, args.N, args.K, args.E)
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    w_t = _dtype_from_args(args, 'w_dtype', 'c10::Float8_e4m3fn')
    scl_t = _dtype_from_args(args, 'scale_dtype', 'float')
    event = {
        'name': 'sglang_profiler::batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_558',
        'args': {
            'Input Dims': [
                [
                    E,
                    M,
                    K],
                [
                    E,
                    N,
                    K],
                [],
                [],
                []],
            'Input type': [
                in_t,
                w_t,
                scl_t,
                '',
                ''] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_fused_flatten_fp8_group_quant(args):
    '''Perf model for sglang_profiler::quant_fused_flatten_fp8_group_quant_37.'''
    mod = _load_dsv3_extension()
    cls = mod.sglang_fused_flatten_fp8_group_quant
    N = args.N
    M = args.M
    if not getattr(args, 'group_size', None):
        getattr(args, 'group_size', None)
    group_size = 128
    if N % group_size != 0:
        raise ValueError(f'''N ({N}) must be divisible by group_size ({group_size})''')
    N2 = group_size
    N1 = N // group_size
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    event = {
        'name': 'sglang_profiler::quant_fused_flatten_fp8_group_quant_37',
        'args': {
            'Input Dims': [
                [
                    M,
                    N1,
                    N2]],
            'Input type': [
                in_t],
            'Input Strides': [
                [
                    N,
                    N2,
                    1]] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_fused_qk_rope_cat_and_cache_mla(args):
    '''Perf model for aiter::fused_qk_rope_cat_and_cache_mla.'''
    mod = _load_dsv3_extension()
    cls = mod.aiter_fused_qk_rope_cat_and_cache_mla
    T = args.M
    if not getattr(args, 'num_heads_q', None):
        getattr(args, 'num_heads_q', None)
    H_q = 16
    if not getattr(args, 'head_dim', None):
        getattr(args, 'head_dim', None)
    head_dim = 576
    if not getattr(args, 'group_size', None):
        getattr(args, 'group_size', None)
    D_pe = 64
    D_lora = head_dim - D_pe
    KH = 1
    num_kv_cache_tokens = max(T + 4096, 8192)
    max_pos = max(T * 16, 4096)
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::BFloat16')
    event = {
        'name': 'aiter::fused_qk_rope_cat_and_cache_mla',
        'args': {
            'Input Dims': [
                [
                    T,
                    H_q,
                    D_lora],
                [
                    T,
                    H_q,
                    D_pe],
                [
                    T,
                    KH,
                    D_lora],
                [
                    T,
                    KH,
                    D_pe],
                [
                    num_kv_cache_tokens,
                    KH,
                    D_lora + D_pe],
                [
                    T],
                [
                    T],
                [
                    max_pos,
                    1,
                    1,
                    D_pe // 2],
                [
                    max_pos,
                    1,
                    1,
                    D_pe // 2],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []],
            'Input type': [
                in_t] * 5 + [
                'long int',
                'long int',
                in_t,
                in_t,
                'float',
                'Scalar',
                'Scalar',
                'Scalar',
                '',
                '',
                '',
                'Scalar'],
            'Concrete Inputs': [
                ''] * 10 + [
                'False',
                '0',
                'True',
                '',
                '',
                '',
                '0'] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_mla_prefill_ps_asm_fwd(args):
    '''Perf model for aiter::mla_prefill_ps_asm_fwd.

    Reads ``--seq_len`` (per-batch ctx), ``--E`` (batch size), ``--num_heads_q``,
    ``--head_dim`` (qk_head_dim; v_head_dim = head_dim - 64).
    '''
    mod = _load_dsv3_extension()
    cls = mod.aiter_mla_prefill_ps_asm_fwd
    seq_len = args.seq_len
    if not getattr(args, 'E', None):
        getattr(args, 'E', None)
    batch = 16
    if not getattr(args, 'num_heads_q', None):
        getattr(args, 'num_heads_q', None)
    H = 16
    if not getattr(args, 'head_dim', None):
        getattr(args, 'head_dim', None)
    d_qk = 192
    d_v = d_qk - 64
    N_total = seq_len * batch
    in_t = _dtype_from_args(args, 'in_dtype', 'c10::Float8_e4m3fn')
    event = {
        'name': 'aiter::mla_prefill_ps_asm_fwd',
        'args': {
            'Input Dims': [
                [
                    N_total,
                    H,
                    d_qk],
                [
                    N_total,
                    H,
                    d_qk],
                [
                    N_total,
                    H,
                    d_v],
                [
                    batch + 1],
                [
                    batch + 1],
                [
                    N_total],
                [
                    batch * 16 + 1],
                [
                    batch * 160,
                    8],
                [],
                [],
                [],
                [
                    batch * 300,
                    H,
                    d_v],
                [
                    batch * 300,
                    H],
                [
                    N_total,
                    H,
                    d_v],
                [],
                [],
                []],
            'Input type': [
                in_t] * 3 + [
                'int'] * 5 + [
                'Scalar'] * 3 + [
                'float',
                'float',
                'c10::BFloat16',
                'float',
                'float',
                'float'],
            'Concrete Inputs': [
                ''] * 8 + [
                str(seq_len),
                '0.13523377886088009',
                'True'] + [
                ''] * 6 } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_mla_reduce_v1(args):
    '''Perf model for aiter::mla_reduce_v1.

    Shape parameters match dsv3_mla_prefill_ps_asm_fwd (planner is shared).
    Partial-output rows are over-allocated by the persistent scheduler; we use
    the same conservative count (~2.4x the final tokens) the trace exhibits.
    '''
    mod = _load_dsv3_extension()
    cls = mod.aiter_mla_reduce_v1
    seq_len = args.seq_len
    if not getattr(args, 'E', None):
        getattr(args, 'E', None)
    batch = 16
    if not getattr(args, 'num_heads_q', None):
        getattr(args, 'num_heads_q', None)
    H = 16
    if not getattr(args, 'head_dim', None):
        getattr(args, 'head_dim', None)
    d_qk = 192
    d_v = d_qk - 64
    N_total = seq_len * batch
    n_partial = max(N_total * 2 + 4000, N_total + 100)
    event = {
        'name': 'aiter::mla_reduce_v1',
        'args': {
            'Input Dims': [
                [
                    n_partial,
                    H,
                    d_v],
                [
                    n_partial,
                    H],
                [
                    batch * 5 + 1],
                [
                    batch * 5,
                    2],
                [
                    n_partial // 100],
                [],
                [
                    N_total,
                    H,
                    d_v],
                [
                    N_total,
                    H]],
            'Input type': [
                'float',
                'float',
                'int',
                'int',
                'int',
                'Scalar',
                'c10::BFloat16',
                'float'],
            'Concrete Inputs': [
                '',
                '',
                '',
                '',
                '',
                '256',
                '',
                ''] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv3_fused_append_shared_experts(args):
    '''Perf model for sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts_456.'''
    mod = _load_dsv3_extension()
    cls = mod.sglang_fused_append_shared_experts
    M = args.M
    if not getattr(args, 'K', None):
        getattr(args, 'K', None)
    K = 8
    event = {
        'name': 'sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts_456',
        'args': {
            'Input Dims': [
                [
                    M,
                    K],
                [
                    M,
                    K]],
            'Input type': [
                'int',
                'float'] } }
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_gemm_afp4wfp4(args):
    '''Perf model for ``aiter::gemm_afp4wfp4_`` (MXFP4 GEMM).

    Inputs are FP4 (packed as ``unsigned char`` in the trace). Output is BF16.
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import gemm_afp4wfp4
    K = args.K
    N = args.N
    M = args.M
    SCALE_GROUP_SIZE = 32
    event = {
        'name': 'aiter::gemm_afp4wfp4_',
        'args': {
            'Input Dims': [
                [
                    M,
                    K // 2],
                [
                    N,
                    K // 2],
                [
                    M,
                    K // SCALE_GROUP_SIZE],
                [
                    N,
                    K // SCALE_GROUP_SIZE],
                [],
                [
                    M,
                    N],
                [],
                []],
            'Input type': [
                'unsigned char',
                'unsigned char',
                'unsigned char',
                'unsigned char',
                'Scalar',
                'c10::BFloat16',
                '',
                'Scalar'],
            'Input Strides': [
                [
                    K // 2,
                    1],
                [
                    K // 2,
                    1],
                [
                    K // SCALE_GROUP_SIZE,
                    1],
                [
                    K // SCALE_GROUP_SIZE,
                    1],
                [],
                [
                    N,
                    1],
                [],
                []],
            'Concrete Inputs': [
                '',
                '',
                '',
                '',
                '15',
                '',
                '',
                'False'] },
        'kernel_names': [
            'placeholder_kernel'],
        'kernel_details': [
            {
                'name': 'placeholder_kernel' }] }
    model = gemm_afp4wfp4(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_rope_cached_positions_2c_fwd_impl(args):
    '''Perf model for ``aiter::rope_cached_positions_2c_fwd_impl``.

    Inputs:
      Input Dims[2] = input_x  (s, b, H_q,  d) BF16
      Input Dims[3] = input_y  (s, b, H_kv, d) BF16
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import aiter_rope_cached_positions_2c_fwd_impl
    M = args.M
    if not getattr(args, 'num_heads_q', None):
        getattr(args, 'num_heads_q', None)
    H_q = 16
    if not getattr(args, 'num_heads_kv', None):
        getattr(args, 'num_heads_kv', None)
    H_kv = 1
    if not getattr(args, 'head_dim', None):
        getattr(args, 'head_dim', None)
    d = 64
    d_cs = d // 2
    max_pos = max(M * 16, 4096)
    event = {
        'name': 'aiter::rope_cached_positions_2c_fwd_impl',
        'args': {
            'Input Dims': [
                [
                    1,
                    M,
                    H_q,
                    d],
                [
                    1,
                    M,
                    H_kv,
                    d],
                [
                    1,
                    M,
                    H_q,
                    d],
                [
                    1,
                    M,
                    H_kv,
                    d],
                [
                    max_pos,
                    1,
                    1,
                    d_cs],
                [
                    max_pos,
                    1,
                    1,
                    d_cs],
                [
                    1,
                    M],
                [],
                [],
                []],
            'Input type': [
                'c10::BFloat16',
                'c10::BFloat16',
                'c10::BFloat16',
                'c10::BFloat16',
                'c10::BFloat16',
                'c10::BFloat16',
                'long int',
                'Scalar',
                'Scalar',
                'Scalar'] } }
    model = aiter_rope_cached_positions_2c_fwd_impl(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_fused_flatten_mxfp4_quant(args):
    '''Perf model for ``sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant``.'''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.perf_model_extensions import fused_flatten_mxfp4_quant
    M = args.M
    N = args.N
    if not getattr(args, 'group_size', None):
        getattr(args, 'group_size', None)
    group_size = 128
    N2 = group_size
    N1 = N // group_size
    event = {
        'name': 'sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant',
        'args': {
            'Input Dims': [
                [
                    M,
                    N1,
                    N2]],
            'Input type': [
                'c10::BFloat16'],
            'Input Strides': [
                [
                    N1 * N2,
                    N2,
                    1]] } }
    model = fused_flatten_mxfp4_quant(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_fused_rms_mxfp4_quant(args):
    '''Perf model for ``sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant``.

    Single-input (no x2, no res1) variant, matching the simplest call shape:
        x1:        (M, N1) BF16
        x1_weight: (N1,)   BF16
    '''
    _ensure_tracelens_importable()
    from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import fused_rms_mxfp4_quant
    N = args.N
    M = args.M
    event = {
        'name': 'sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant',
        'args': {
            'Input Dims': [
                [
                    M,
                    N],
                [
                    N],
                []],
            'Input type': [
                'c10::BFloat16',
                'c10::BFloat16',
                'Scalar'],
            'Input Strides': [
                [
                    N,
                    1],
                [
                    1],
                []] } }
    model = fused_rms_mxfp4_quant(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_from_event(trace_name, input_dims, input_types, input_strides, concrete_inputs, attn_params = (None,)):
    '''Instantiate the perf model directly from CSV columns.

    Constructs an event dict matching TraceLens internal format and looks up the
    perf model class via op_to_perf_model_class_map.
    '''
    _ensure_tracelens_importable()
    dims_as_lists = _tuple_to_list(input_dims)
    types_as_lists = list(input_types) if input_types else []
    strides_as_lists = _tuple_to_list(input_strides) if input_strides else []
    concrete_as_lists = list(concrete_inputs) if concrete_inputs else []
    event = {
        'name': trace_name,
        'args': {
            'Input Dims': dims_as_lists,
            'Input type': types_as_lists } }
    if strides_as_lists:
        event['args']['Input Strides'] = strides_as_lists
    if concrete_as_lists:
        event['args']['Concrete Inputs'] = concrete_as_lists
    if attn_params:
        c_sq = attn_params.get('c_sq', 0)
        c_sk = attn_params.get('c_sk', 0)
        c_sqsq = attn_params.get('c_sqsq', 0)
        c_sqsk = attn_params.get('c_sqsk', 0)
        g_sq = attn_params.get('g_sq', 0)
        g_sk = attn_params.get('g_sk', 0)
        g_sqsq = attn_params.get('g_sqsq', 0)
        g_sqsk = attn_params.get('g_sqsk', 0)
        
        def _safe_int(v):
            if v is None or str(v) in ('nan', ''):
                return 0
            return int(float(v))

        c_sq = _safe_int(c_sq)
        c_sk = _safe_int(c_sk)
        c_sqsq = _safe_int(c_sqsq)
        c_sqsk = _safe_int(c_sqsk)
        g_sq = _safe_int(g_sq)
        g_sk = _safe_int(g_sk)
        g_sqsq = _safe_int(g_sqsq)
        g_sqsk = _safe_int(g_sqsk)
        if any((v != 0 for v in (g_sq, g_sk, g_sqsq, g_sqsk))):
            annotation = f'attn_csq_{c_sq}_csk_{c_sk}_csqsq_{c_sqsq}_csqsk_{c_sqsk}_gsq_{g_sq}_gsk_{g_sk}_gsqsq_{g_sqsq}_gsqsk_{g_sqsk}'
        else:
            annotation = f'attn_0_0_{c_sq}_{c_sqsq}_0_0_0'
        event['annotation'] = annotation
    event['kernel_names'] = [
        'placeholder_kernel']
    event['kernel_details'] = [
        {
            'name': 'placeholder_kernel' }]
    try:
        from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
    except ImportError:
        from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import get_pseudo_op_mappings
        op_to_perf_model_class_map = get_pseudo_op_mappings()
    model_cls = op_to_perf_model_class_map.get(trace_name)
    if model_cls is None:
        raise ValueError(f"No perf model class found for trace name '{trace_name}'")
    model = model_cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


# ---------------------------------------------------------------------------
# DSV4 (SGLang / AITER) triage-extension perf model runners
# ---------------------------------------------------------------------------
_DSV4_EXT_DEFAULT = (
    '/home/devashah/Magpie/results/benchmark_sglang_20260615_154820/torch_trace/'
    'steady_state_16/analysis_output_mixed/perf_report_csvs/'
    'unified_perf_summary_triage_extension.py'
)
_DSV4_EXT_CACHE = {}

_BF16 = 'c10::BFloat16'
_FP8 = 'c10::Float8_e4m3fn'
_E8M0 = 'c10::Float8_e8m0fnu'


def _load_dsv4_extension():
    '''Load (and cache) the DSV4 triage extension module + its name->class map.'''
    path = _os.environ.get('DSV4_TRIAGE_EXTENSION', _DSV4_EXT_DEFAULT)
    cached = _DSV4_EXT_CACHE.get(path)
    if cached is not None:
        return cached
    _ensure_tracelens_importable()
    spec = _ilu.spec_from_file_location('dsv4_triage_extension', path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load DSV4 triage extension from '{path}'. "
            "Set DSV4_TRIAGE_EXTENSION to the triage extension path."
        )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DSV4_EXT_CACHE[path] = mod
    return mod


def _dsv4_eval(name, event):
    '''Instantiate the extension perf-model class for ``name`` and evaluate.'''
    mod = _load_dsv4_extension()
    cls = mod.perf_model_extension[name]
    model = cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def run_perf_model_dsv4_mhc_pre_gemm_sqrsum(args):
    M = args.M
    C = args.N
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
    K = hc_mult * C                              # 28672
    split_k = getattr(args, 'split_k', None) or 16
    event = {
        'name': 'aiter::mhc_pre_gemm_sqrsum',
        'args': {
            'Input Dims': [[split_k, M, hc_mult3], [split_k, M],
                           [M, hc_mult, C], [hc_mult3, K], []],
            'Input type': ['float', 'float', _BF16, 'float', 'Scalar'],
        },
    }
    return _dsv4_eval('aiter::mhc_pre_gemm_sqrsum', event)


def run_perf_model_dsv4_mhc_pre_big_fuse(args):
    M = args.M
    C = args.N
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    split_k = getattr(args, 'split_k', None) or 16
    event = {
        'name': 'aiter::mhc_pre_big_fuse',
        'args': {
            'Input Dims': [[M, hc_mult, 1], [M, hc_mult, hc_mult], [M, C],
                           [split_k, M, hc_mult3], [split_k, M], [3], [hc_mult3],
                           [M, hc_mult, C], [], [], [], [], []],
            'Input type': ['float', 'float', _BF16, 'float', 'float', 'float',
                           'float', _BF16, 'Scalar', 'Scalar', 'Scalar', 'Scalar', 'Scalar'],
            'Concrete Inputs': ['', '', '', '', '', '', '', '',
                                '1e-06', '1e-06', '1e-06', '2.', '20'],
        },
    }
    return _dsv4_eval('aiter::mhc_pre_big_fuse', event)


def run_perf_model_dsv4_mhc_post(args):
    M = args.M
    C = args.N
    hc_mult = 4
    event = {
        'name': 'aiter::mhc_post',
        'args': {
            'Input Dims': [[M, hc_mult, C], [M, C], [M, hc_mult, C],
                           [M, hc_mult], [M, hc_mult, hc_mult]],
            'Input type': [_BF16, _BF16, _BF16, 'float', 'float'],
        },
    }
    return _dsv4_eval('aiter::mhc_post', event)


def run_perf_model_dsv4_pa_sparse_prefill_opus(args):
    N = args.M
    H = getattr(args, 'num_heads_q', None) or 32
    D = getattr(args, 'head_dim', None) or 512
    nnz_prefix = 2095488
    nnz_extend = 232832
    total_pages = 329728
    total_tokens = N
    event = {
        'name': 'aiter::pa_sparse_prefill_opus_fwd',
        'args': {
            'Input Dims': [[N, H, D], [total_pages, D], [nnz_prefix], [N + 1],
                           [N, D], [nnz_extend], [N + 1], [H], [N, H, D], []],
            'Input type': [_BF16, _BF16, 'int', 'int', _BF16, 'int', 'int',
                           'float', _BF16, 'Scalar'],
        },
    }
    return _dsv4_eval('aiter::pa_sparse_prefill_opus_fwd', event)


def run_perf_model_dsv4_opus_gemm_a16w16(args):
    M, N, K = args.M, args.N, args.K
    event = {
        'name': 'aiter::_opus_gemm_a16w16_tune_raw',
        'args': {
            'Input Dims': [[1, M, K], [1, N, K], [1, M, N], [], [], []],
            'Input type': [_BF16, _BF16, _BF16, '', 'Scalar', 'Scalar'],
        },
    }
    return _dsv4_eval('aiter::_opus_gemm_a16w16_tune_raw', event)


def run_perf_model_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm(args):
    M, N, K = args.M, args.N, args.K
    sk = (K + 127) // 128
    sn = (N + 127) // 128
    event = {
        'name': 'aiter::_gemm_a8w8_blockscale_bpreshuffle_asm',
        'args': {
            'Input Dims': [[M, K], [N, K], [M, N], [M, sk], [sn, sk],
                           [], [], [], [], [1, N]],
            'Input type': [_FP8, _FP8, _BF16, 'float', 'float', '', 'Scalar', '', 'Scalar', 'float'],
        },
    }
    return _dsv4_eval('aiter::_gemm_a8w8_blockscale_bpreshuffle_asm', event)


def run_perf_model_dsv4_dynamic_per_group_scaled_quant(args):
    M = args.M
    K = args.N
    g = getattr(args, 'group_size', None) or 32
    sk = K // g
    event = {
        'name': 'aiter::dynamic_per_group_scaled_quant',
        'args': {
            'Input Dims': [[M, K], [M, K], [M, sk], [], [], [], []],
            'Input type': [_FP8, _BF16, _E8M0, 'Scalar', 'Scalar', '', 'Scalar'],
            'Input Strides': [[K, 1], [K, 1], [sk, 1], [], [], [], []],
            'Concrete Inputs': ['', '', '', str(g), 'False', '', '1'],
        },
    }
    return _dsv4_eval('aiter::dynamic_per_group_scaled_quant', event)


def run_perf_model_dsv4_topk_softplus(args):
    T = args.M
    E = args.N
    k = getattr(args, 'topk', None) or 6
    event = {
        'name': 'aiter::topk_softplus',
        'args': {
            'Input Dims': [[T, k], [T, k], [T, E], [E], [], [], []],
            'Input type': ['float', 'int', _BF16, _BF16, 'Scalar', 'Scalar', ''],
            'Concrete Inputs': ['', '', '', '', 'True', '2.5', ''],
        },
    }
    return _dsv4_eval('aiter::topk_softplus', event)


def run_perf_model_dsv4_fused_dynamic_mx_quant_moe_sort(args):
    R = args.M
    C = args.N
    g = getattr(args, 'group_size', None) or 32
    E = getattr(args, 'E', None) or 384
    topk = getattr(args, 'topk', None) or 6
    block = getattr(args, 'block_m', None) or 32
    sk = C // g
    # sorted-id buffer size mirrors aiter.moe_sorting: R*topk + E*block - R
    n_sorted = R * topk + E * block - R
    s_pad = ((n_sorted + 31) // 32) * 32
    event = {
        'name': 'aiter::fused_dynamic_mx_quant_moe_sort_hip',
        'args': {
            'Input Dims': [[R, C], [s_pad, sk], [R, C], [n_sorted], [2], [], [], []],
            'Input type': [_FP8, _E8M0, _BF16, 'int', 'int', 'Scalar', 'Scalar', 'Scalar'],
            'Input Strides': [[C, 1], [sk, 1], [C, 1], [1], [1], [], [], []],
            'Concrete Inputs': ['', '', '', '', '', str(g), str(g), str(g)],
        },
    }
    return _dsv4_eval('aiter::fused_dynamic_mx_quant_moe_sort_hip', event)

