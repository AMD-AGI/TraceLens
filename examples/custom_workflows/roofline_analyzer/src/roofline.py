import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from io import BytesIO

def calculate_ridge_points(config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate the arithmetic intensity ridge points.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (AI_ridge_mtf, AI_ridge_maf)
    """
    max_memory_bandwidth = config['accelerator']['max_memory_bandwidth']
    max_compute_teraflops = config['accelerator']['max_compute_teraflops']
    max_achievable_teraflops = config['accelerator']['max_achievable_teraflops']
    
    # Ridge point = Peak Compute / Memory Bandwidth
    ai_ridge_mtf = max_compute_teraflops / max_memory_bandwidth
    ai_ridge_maf = max_achievable_teraflops / max_memory_bandwidth
    
    return ai_ridge_mtf, ai_ridge_maf

def calculate_rooflines(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Calculate the roofline model values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with x-values and y-values for each roofline
    """
    max_memory_bandwidth = config['accelerator']['max_memory_bandwidth']
    max_compute_teraflops = config['accelerator']['max_compute_teraflops']
    max_achievable_teraflops = config['accelerator']['max_achievable_teraflops']
    
    # Create x-values (operational intensity) range
    x_values = np.logspace(-2, 3, 1000)  # From 0.01 to 1000 FLOPS/byte
    
    # Calculate memory roofline: y = x * bandwidth
    memory_roof = x_values * max_memory_bandwidth
    
    # Calculate compute rooflines (horizontal lines)
    theoretical_roof = np.full_like(x_values, max_compute_teraflops)
    achievable_roof = np.full_like(x_values, max_achievable_teraflops)
    
    # Create the combined rooflines (min of memory and compute)
    combined_theoretical = np.minimum(memory_roof, theoretical_roof)
    combined_achievable = np.minimum(memory_roof, achievable_roof)
    
    return {
        'x_values': x_values,
        'memory_roof': memory_roof,
        'theoretical_roof': theoretical_roof,
        'achievable_roof': achievable_roof,
        'combined_theoretical': combined_theoretical,
        'combined_achievable': combined_achievable
    }

def create_roofline_plot(
    df: pd.DataFrame, 
    rooflines: Dict[str, np.ndarray],
    config: Dict[str, Any],
    ai_ridge_mtf: float,
    ai_ridge_maf: float
) -> plt.Figure:
    """
    Generate the roofline plot with kernel data points.
    
    Args:
        df: DataFrame with kernel data
        rooflines: Dictionary with roofline data
        config: Configuration dictionary
        ai_ridge_mtf: The arithmetic intensity ridge point for max theoretical flops
        ai_ridge_maf: The arithmetic intensity ridge point for max achievable flops
        
    Returns:
        Matplotlib figure object with the roofline plot
    """
    # Get column names from config
    flops_per_byte_col = config['excel']['flops_per_byte_column']
    performance_col = config['excel']['performance_column']
    kernel_name_col = config['excel'].get('kernel_name_column', None)
    
    # Create figure and axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot rooflines
    ax.loglog(rooflines['x_values'], rooflines['memory_roof'], 'b-', 
             label=f"Memory Bandwidth ({config['accelerator']['max_memory_bandwidth']} TB/s)")
    
    ax.loglog(rooflines['x_values'], rooflines['theoretical_roof'], 'r-', 
             label=f"Max Theoretical ({config['accelerator']['max_compute_teraflops']} TFLOPS)")
    
    ax.loglog(rooflines['x_values'], rooflines['achievable_roof'], 'g-', 
             label=f"Max Achievable ({config['accelerator']['max_achievable_teraflops']} TFLOPS)")
    
    # Plot ridge points
    ax.axvline(x=ai_ridge_mtf, color='r', linestyle='--', alpha=0.5, 
              label=f'AI Ridge MTF ({ai_ridge_mtf:.2f})')
    
    ax.axvline(x=ai_ridge_maf, color='g', linestyle='--', alpha=0.5, 
              label=f'AI Ridge MAF ({ai_ridge_maf:.2f})')
    
    # Plot kernel points
    ax.scatter(df[flops_per_byte_col], df[performance_col], c='black', s=50, alpha=0.7)
    
    # Add annotations for each kernel using the kernel_name_col (should be 'ID')
    if kernel_name_col and kernel_name_col in df.columns:
        for i, row in df.iterrows():
            kernel_name = row[kernel_name_col]
            ax.annotate(str(kernel_name),
                        (row[flops_per_byte_col], row[performance_col]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha='left',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)')
    ax.set_ylabel('Performance (TFLOPS/s)')
    ax.set_title('Roofline Analysis')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    # Tight layout for better spacing
    fig.tight_layout()
    
    return fig

def plot_roofline(
    df: pd.DataFrame, 
    rooflines: Dict[str, np.ndarray],
    config: Dict[str, Any],
    ai_ridge_mtf: float,
    ai_ridge_maf: float,
    output_file: str = None
) -> plt.Figure:
    """
    Generate and optionally save the roofline plot.
    
    Args:
        df: DataFrame with kernel data
        rooflines: Dictionary with roofline data
        config: Configuration dictionary
        ai_ridge_mtf: The arithmetic intensity ridge point for max theoretical flops
        ai_ridge_maf: The arithmetic intensity ridge point for max achievable flops
        output_file: Optional file path to save the plot
        
    Returns:
        Matplotlib figure object with the roofline plot
    """
    fig = create_roofline_plot(df, rooflines, config, ai_ridge_mtf, ai_ridge_maf)
    
    # Save if output file is specified
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Roofline plot saved to {output_file}")
    
    return fig