###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Template for adding new operation test harnesses to ``validate_perf_model``.

Copy this file to ``tests/<category>.py``, rename functions/metadata, and
follow the instructions in each section.

---------------------------------------------------------------------------
THREE INVOCATION PATTERNS
---------------------------------------------------------------------------
Kernels in this framework come from three sources; pick the right pattern:

  1. ``aiter`` (Python package call)
       import aiter
       aiter.<fn>(inputs...)                   # or aiter.ops.<sub>.<fn>

  2. ``vLLM`` (torch.ops dispatch)
       import vllm._aiter_ops   # or another vllm side-effect import
       torch.ops.vllm.<fn>(inputs...)

  3. ``SGLang`` (sgl_kernel package call)
       import sgl_kernel
       sgl_kernel.<fn>(inputs...)              # or sgl_kernel.ops.<fn>

Use the harness skeleton that matches your kernel's source, then register
it in ``OP_METADATA`` and in ``tests/_runner.py``.

---------------------------------------------------------------------------
OP_METADATA FORMAT
---------------------------------------------------------------------------
Each harness file should expose a top-level ``OP_METADATA`` dict whose keys
are the OP_REGISTRY / _runner.py op-names and whose values contain:

    {
        "fn":           <callable>  # the test_<op> function in this file
        "category":     str         # base extension class category string
        "description":  str         # one-line description for discovery output
        "dtypes":       list[str]   # all supported dtype strings
        "defaults":     dict        # default kwargs for generate_test_cases --auto
        "required_args": list[str]  # dimension args that must be set explicitly
    }

``generate_test_cases.py`` reads OP_METADATA to auto-generate CSV test cases;
``run_validation.py`` reads it for bulk discovery.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


# ---------------------------------------------------------------------------
# Pattern 1: aiter invocation
# ---------------------------------------------------------------------------

def test_template_aiter_op(
    M, N,
    in_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter.<fn_name>`` template.

    Replace ``<fn_name>`` with the actual aiter function, update the
    docstring dtypes/shapes, and adjust tensor construction accordingly.

    Parameters
    ----------
    M : int
        Batch / token dimension.
    N : int
        Hidden / feature dimension.
    in_dtype : {"bf16", "fp16", "fp8", "i8"}
        Input activation dtype.
    out_dtype : {"bf16", "fp16", "fp8"}
        Output dtype.
    """
    import torch
    import aiter  # noqa: F401 (replace with correct aiter sub-import as needed)

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    device = "cuda"

    print(f"test: template_aiter_op M={M} N={N} in={in_dtype} out={out_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    # Warmup
    for _ in range(num_warmup):
        # REPLACE: aiter.<fn_name>(out, inp)
        pass
    torch.cuda.synchronize()

    # Measured iteration (rocprofv3 records this single dispatch)
    print("test: measured iteration...", flush=True)
    # REPLACE: aiter.<fn_name>(out, inp)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# Pattern 2: vLLM torch.ops dispatch
# ---------------------------------------------------------------------------

def test_template_vllm_op(
    M, N,
    in_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``torch.ops.vllm.<fn_name>`` template.

    Parameters
    ----------
    M : int
    N : int
    in_dtype : {"bf16", "fp16"}
    out_dtype : {"bf16", "fp16"}
    """
    import torch
    # Side-effect import to register torch.ops.vllm kernels:
    import vllm._aiter_ops  # noqa: F401  (or: from vllm.model_executor... import something)

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    device = "cuda"

    print(f"test: template_vllm_op M={M} N={N} in={in_dtype} out={out_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    fn = torch.ops.vllm  # REPLACE: torch.ops.vllm.<fn_name>

    for _ in range(num_warmup):
        # REPLACE: fn(inp, out)
        pass
    torch.cuda.synchronize()

    print("test: measured iteration...", flush=True)
    # REPLACE: fn(inp, out)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# Pattern 3: SGLang sgl_kernel invocation
# ---------------------------------------------------------------------------

def test_template_sglang_op(
    M, N,
    in_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``sgl_kernel.<fn_name>`` template.

    Parameters
    ----------
    M : int
    N : int
    in_dtype : {"bf16", "fp16"}
    out_dtype : {"bf16", "fp16"}
    """
    import torch
    import sgl_kernel  # noqa: F401  (or: from sgl_kernel import <fn_name>)

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    device = "cuda"

    print(f"test: template_sglang_op M={M} N={N} in={in_dtype} out={out_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    for _ in range(num_warmup):
        # REPLACE: sgl_kernel.<fn_name>(inp, out)
        pass
    torch.cuda.synchronize()

    print("test: measured iteration...", flush=True)
    # REPLACE: sgl_kernel.<fn_name>(inp, out)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# OP_METADATA — register harnesses for generate_test_cases and discovery
# ---------------------------------------------------------------------------

#: Metadata table for all ops defined in this harness file.
#: Keys must match the OP_REGISTRY / _runner.py op-names exactly.
OP_METADATA: dict = {
    # --- aiter pattern example ---
    "template_aiter_op": {
        "fn":           test_template_aiter_op,
        "category":     "UnaryElementwise",    # base extension class .category
        "description":  "Template: aiter elementwise op",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 128, "N": 4096, "in_dtype": "bf16", "out_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    # --- vLLM pattern example ---
    "template_vllm_op": {
        "fn":           test_template_vllm_op,
        "category":     "GroupQuant",
        "description":  "Template: vLLM torch.ops dispatch",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 128, "N": 4096, "in_dtype": "bf16", "out_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    # --- SGLang pattern example ---
    "template_sglang_op": {
        "fn":           test_template_sglang_op,
        "category":     "FusedRoPE",
        "description":  "Template: sgl_kernel dispatch",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 128, "N": 4096, "in_dtype": "bf16", "out_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
}
