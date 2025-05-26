import typer
import tomli
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Roofline Analysis Tool")
console = Console()

def read_toml_config(config_path: str) -> Dict[str, Any]:
    """
    Read and parse the TOML configuration file.
    
    Args:
        config_path: Path to the TOML config file
        
    Returns:
        Dictionary with configuration values
    """
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        
        # Validate required config sections and keys
        required_sections = {
            "accelerator": ["max_memory_bandwidth", "max_compute_teraflops", "max_achievable_teraflops"],
            "excel": ["sheet_name", "flops_per_byte_column", "performance_column"],
            "output": ["prefix"]
        }
        
        for section, keys in required_sections.items():
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
            
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"Missing required key in config[{section}]: {key}")
        
        return config
    except Exception as e:
        raise RuntimeError(f"Error reading config file: {e}")

def display_summary(
    df_orig: Any, 
    df_result: Any, 
    output_file: str,
    config: Dict[str, Any],
    ai_ridge_mtf: float,
    ai_ridge_maf: float
) -> None:
    """
    Display a summary of the analysis.
    
    Args:
        df_orig: Original DataFrame
        df_result: Resulting DataFrame after analysis
        output_file: Path to the output Excel file
        config: Configuration dictionary
        ai_ridge_mtf: The arithmetic intensity ridge point for max theoretical flops
        ai_ridge_maf: The arithmetic intensity ridge point for max achievable flops
    """
    console.print(f"\n[bold green]Roofline Analysis Complete[/bold green]")
    
    # Hardware information
    hw_table = Table(title="Hardware Configuration")
    hw_table.add_column("Parameter", style="cyan")
    hw_table.add_column("Value", style="green")
    
    hw_table.add_row("Max Memory Bandwidth", f"{config['accelerator']['max_memory_bandwidth']} TB/s")
    hw_table.add_row("Max Compute", f"{config['accelerator']['max_compute_teraflops']} TFLOPS")
    hw_table.add_row("Max Achievable Compute", f"{config['accelerator']['max_achievable_teraflops']} TFLOPS")
    
    console.print(hw_table)
    
    # Analysis information
    analysis_table = Table(title="Analysis Results")
    analysis_table.add_column("Parameter", style="cyan")
    analysis_table.add_column("Value", style="green")
    
    analysis_table.add_row("Kernels Analyzed", str(len(df_orig)))
    analysis_table.add_row("Compute Bound Kernels", 
                          str(len(df_result[df_result['bound_type_maf'] == 'compute'])))
    analysis_table.add_row("Memory Bound Kernels", 
                          str(len(df_result[df_result['bound_type_maf'] == 'memory'])))
    analysis_table.add_row("AI Ridge MTF", f"{ai_ridge_mtf:.4f}")
    analysis_table.add_row("AI Ridge MAF", f"{ai_ridge_maf:.4f}")
    
    console.print(analysis_table)
    
    # Output information
    console.print(f"\n[bold]Output file:[/bold] {output_file}")