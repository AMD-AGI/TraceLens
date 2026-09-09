"""TraceLens ModelUtils: CPU-only LLM architecture analysis and model parsing.

Two graph-building backends live side by side here:

- ``torch_trace`` (default): traces the model on the meta device with PyTorch
  (``named_modules()`` hierarchy + per-module ``torch.fx`` tracing + forward
  hooks) to build the Model Explorer graph. See ``build_graph``.
- AST backend (``ast_analyze`` / ``extract`` / ``computation_graph`` / ...):
  the original static-analysis pipeline that parses a model's
  ``modeling_*.py`` source with Python's ``ast`` module instead of executing
  it. Kept as a switchable fallback (e.g. for checkpoints that can't be
  instantiated/traced, or for offline source-only inspection). See
  ``load_model_spec`` / ``load_architecture``.

Both backends are wired into the CLI via ``--backend {torch,ast}``
(see ``TraceLens.Visualizer.model_explorer_export.cli``).
"""

from TraceLens.ModelUtils.ast_analyze import (
    analyze_source,
    analyze_sources,
    dump_ast,
    parse_python_ast,
)
from TraceLens.ModelUtils.blocks import BlockComponent, CodeAnalysis
from TraceLens.ModelUtils.extract import (
    ArchitectureSpec,
    architecture_section_trees,
    dump_model_ast,
    load_architecture,
)
from TraceLens.ModelUtils.github import (
    fetch_github_source,
    is_github_url,
    parse_github_url,
)
from TraceLens.ModelUtils.model_graph import (
    ModelGraph,
    build_architecture_model_graphs,
    build_model_graph,
    save_architecture_model_graphs,
    save_model_graph,
)
from TraceLens.ModelUtils.loader import (
    build_detailed_basic_ops,
    load_model_spec,
    resolve_checkpoint_arg,
)
from TraceLens.ModelUtils.shape_inference import (
    ShapeInferencer,
    ShapeContext,
    TensorSpec,
    build_operator_export,
    save_operator_export,
)
from TraceLens.ModelUtils.torch_trace import build_graph

__all__ = [
    "ArchitectureSpec",
    "BlockComponent",
    "CodeAnalysis",
    "ModelGraph",
    "ShapeContext",
    "ShapeInferencer",
    "TensorSpec",
    "analyze_source",
    "analyze_sources",
    "architecture_section_trees",
    "build_architecture_model_graphs",
    "build_detailed_basic_ops",
    "build_graph",
    "build_model_graph",
    "build_operator_export",
    "dump_ast",
    "dump_model_ast",
    "fetch_github_source",
    "is_github_url",
    "load_architecture",
    "load_model_spec",
    "parse_github_url",
    "parse_python_ast",
    "resolve_checkpoint_arg",
    "save_architecture_model_graphs",
    "save_model_graph",
    "save_operator_export",
]
__version__ = "0.5.0"
