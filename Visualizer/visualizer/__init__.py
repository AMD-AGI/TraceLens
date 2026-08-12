"""TraceLens Visualizer: CPU-only LLM architecture diagram generator."""

from visualizer.ast_analyze import analyze_source, analyze_sources, dump_ast, parse_python_ast
from visualizer.blocks import BlockComponent, CodeAnalysis
from visualizer.extract import ArchitectureSpec, dump_model_ast, load_architecture
from visualizer.github import fetch_github_source, is_github_url, parse_github_url
from visualizer.model_graph import (
    ModelGraph,
    build_architecture_model_graphs,
    build_model_graph,
    save_architecture_model_graphs,
    save_model_graph,
)
from visualizer.render import render_diagram

__all__ = [
    "ArchitectureSpec",
    "BlockComponent",
    "CodeAnalysis",
    "ModelGraph",
    "analyze_source",
    "analyze_sources",
    "build_architecture_model_graphs",
    "build_model_graph",
    "dump_ast",
    "dump_model_ast",
    "fetch_github_source",
    "is_github_url",
    "load_architecture",
    "parse_github_url",
    "parse_python_ast",
    "render_diagram",
    "save_architecture_model_graphs",
    "save_model_graph",
]
__version__ = "0.3.0"
