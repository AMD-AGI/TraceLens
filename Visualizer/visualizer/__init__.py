"""TraceLens Visualizer: CPU-only LLM architecture diagram generator."""

from visualizer.ast_analyze import analyze_source, analyze_sources, dump_ast, parse_python_ast
from visualizer.blocks import BlockComponent, CodeAnalysis
from visualizer.extract import ArchitectureSpec, dump_model_ast, load_architecture
from visualizer.github import fetch_github_source, is_github_url, parse_github_url
from visualizer.render import render_diagram

__all__ = [
    "ArchitectureSpec",
    "BlockComponent",
    "CodeAnalysis",
    "analyze_source",
    "analyze_sources",
    "dump_ast",
    "dump_model_ast",
    "fetch_github_source",
    "is_github_url",
    "load_architecture",
    "parse_github_url",
    "parse_python_ast",
    "render_diagram",
]
__version__ = "0.3.0"
