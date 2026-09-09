"""Export TraceLens model graphs to AI Edge Model Explorer format.

Two backends are available (see ``cli.py``'s ``--backend`` flag):

- ``build`` (torch, default): ``build_model_explorer_payload`` traces the
  model with PyTorch on the meta device (``torch_trace.build_graph``).
- ``ast_build`` (ast, backup): ``build_ast_model_explorer_payload`` builds
  the graph from a statically-parsed ``ArchitectureSpec`` without executing
  any model code (``ast_build.build_model_explorer_payload``).
"""

from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload as build_ast_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.build import (
    build_model_explorer_payload,
    save_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.adapter import (
    computation_graph_to_explorer_graph,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import save_viewer_html

__all__ = [
    "build_ast_model_explorer_payload",
    "build_model_explorer_payload",
    "computation_graph_to_explorer_graph",
    "save_model_explorer_payload",
    "save_viewer_html",
]
