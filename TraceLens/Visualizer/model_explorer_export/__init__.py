"""Export TraceLens model graphs to AI Edge Model Explorer format.

The graph is built by the AST backend: ``build_ast_model_explorer_payload``
builds it from a statically-parsed ``ArchitectureSpec`` without executing any
model code (``ast_build.build_model_explorer_payload``). A PyTorch meta-device
pass is used only to fill in shapes the static analysis can't infer.
"""

from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload as build_ast_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.build import (
    save_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.adapter import (
    computation_graph_to_explorer_graph,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import save_viewer_html
from TraceLens.ModelUtils.torch_introspect import (
    build_torch_module_payload,
    payload_from_fx_inventory,
)

__all__ = [
    "build_ast_model_explorer_payload",
    "build_torch_module_payload",
    "computation_graph_to_explorer_graph",
    "payload_from_fx_inventory",
    "save_model_explorer_payload",
    "save_viewer_html",
]
