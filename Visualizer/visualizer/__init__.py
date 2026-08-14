"""TraceLens Visualizer: CPU-only LLM architecture diagram generator."""

from visualizer.ast_analyze import analyze_source, analyze_sources, dump_ast, parse_python_ast
from visualizer.blocks import BlockComponent, CodeAnalysis
from visualizer.extract import ArchitectureSpec, architecture_section_trees, dump_model_ast, load_architecture
from visualizer.github import fetch_github_source, is_github_url, parse_github_url
from visualizer.model_graph import (
    ModelGraph,
    build_architecture_model_graphs,
    build_model_graph,
    save_architecture_model_graphs,
    save_model_graph,
)
from visualizer.loader import build_detailed_basic_ops, load_model_spec, resolve_checkpoint_arg
from visualizer.render import render_diagram
from visualizer.shape_inference import ShapeInferencer, build_operator_export, save_operator_export

__all__ = [
    "ArchitectureSpec",
    "BlockComponent",
    "CodeAnalysis",
    "ModelGraph",
    "ShapeInferencer",
    "analyze_source",
    "analyze_sources",
    "architecture_section_trees",
    "build_architecture_model_graphs",
    "build_detailed_basic_ops",
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
    "render_diagram",
    "resolve_checkpoint_arg",
    "save_architecture_model_graphs",
    "save_model_graph",
    "save_operator_export",
]
__version__ = "0.3.0"
