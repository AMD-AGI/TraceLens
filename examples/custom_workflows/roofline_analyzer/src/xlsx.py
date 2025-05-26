import pandas as pd
import time
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, Any, Tuple

def read_xlsx(file_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Read and parse the Excel file containing GEMM kernels data.
    
    Args:
        file_path: Path to the Excel file
        config: Configuration dictionary from TOML
        
    Returns:
        DataFrame containing kernel data
    """
    sheet_name = config['excel']['sheet_name']
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        required_columns = [
            config['excel']['flops_per_byte_column'],
            config['excel']['performance_column']
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in Excel file: {missing_cols}")
            
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")

def add_analysis_columns(
    df: pd.DataFrame, 
    config: Dict[str, Any],
    ai_ridge_mtf: float,
    ai_ridge_maf: float
) -> pd.DataFrame:
    """
    Add calculated columns to the DataFrame for roofline analysis.
    
    Args:
        df: DataFrame with kernel data
        config: Configuration dictionary
        ai_ridge_mtf: The arithmetic intensity ridge point for max theoretical flops
        ai_ridge_maf: The arithmetic intensity ridge point for max achievable flops
        
    Returns:
        DataFrame with added analysis columns
    """
    # Add ID column as the first column (1-based index)
    df = df.copy()
    df.insert(0, 'ID', range(1, len(df) + 1))
    # Set config to use 'ID' as kernel_name_column for plotting
    config['excel']['kernel_name_column'] = 'ID'
    
    # Column name mappings from config
    flops_per_byte_col = config['excel']['flops_per_byte_column']
    performance_col = config['excel']['performance_column']
    
    # Constants from config
    max_memory_bandwidth = config['accelerator']['max_memory_bandwidth']
    max_achievable_tflops = config['accelerator']['max_achievable_teraflops']
    
    # Calculate memory roofline for each kernel
    df['kernel_memory_roofline'] = df[flops_per_byte_col] * max_memory_bandwidth
    
    # Determine if kernel is memory or compute bound
    df['bound_type_maf'] = df.apply(
        lambda row: "compute" if row[flops_per_byte_col] >= ai_ridge_maf else "memory", 
        axis=1
    )
    
    # Add column for the reference roofline value used for distance calculation
    def get_reference_roofline(row):
        if row['bound_type_maf'] == 'compute':
            return max_achievable_tflops
        else:
            return row['kernel_memory_roofline']
    
    df['reference_roofline'] = df.apply(get_reference_roofline, axis=1)
    
    # Calculate distance to nearest roofline
    def calculate_bound_distance(row):
        # Now we can just use the reference_roofline value
        return row['reference_roofline'] - row[performance_col]
    
    df['bound_distance'] = df.apply(calculate_bound_distance, axis=1)
    
    # Calculate percentage distance
    def calculate_bound_distance_pct(row):
        return (row['bound_distance'] / row['reference_roofline']) * 100
    
    df['bound_distance_pct'] = df.apply(calculate_bound_distance_pct, axis=1)
    
    return df

def export_to_xlsx(
    df_result: pd.DataFrame,
    df_orig: pd.DataFrame,
    config: Dict[str, Any],
    ai_ridge_mtf: float,
    ai_ridge_maf: float,
    plot_figure: plt.Figure = None
) -> str:
    """
    Export the analyzed data to a new Excel file.
    
    Args:
        df_result: DataFrame with analyzed data (with calculated columns)
        df_orig: Original DataFrame without calculated columns
        config: Configuration dictionary
        ai_ridge_mtf: The arithmetic intensity ridge point for max theoretical flops
        ai_ridge_maf: The arithmetic intensity ridge point for max achievable flops
        plot_figure: Optional matplotlib figure to include in the Excel file
        
    Returns:
        Path to the exported file
    """
    timestamp = int(time.time())
    output_file = f"{config['output']['prefix']}-{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Export the analyzed data
        sheet_name = config['excel']['sheet_name']
        df_result.to_excel(writer, sheet_name=f"{sheet_name}_analyzed", index=False)
        
        # Export the original data
        df_orig.to_excel(writer, sheet_name=f"{sheet_name}_original", index=False)
        
        # Create a sheet for scalar values
        scalar_df = pd.DataFrame({
            'Value': [ai_ridge_mtf, ai_ridge_maf]
        }, index=['AI_ridge_mtf', 'AI_ridge_maf'])
        
        scalar_df.to_excel(writer, sheet_name='ScalarValues')
        
        # Include the plot if provided
        if plot_figure:
            # Create a 'Plots' sheet
            workbook = writer.book
            plot_sheet = workbook.create_sheet(title='RooflinePlot')
            
            # Save the figure to a BytesIO object
            img_data = BytesIO()
            plot_figure.savefig(img_data, format='png', dpi=300)
            img_data.seek(0)
            
            # Add the image to the workbook
            from openpyxl.drawing.image import Image
            img = Image(img_data)
            
            # You can adjust the size and position as needed
            img.width = 800
            img.height = 500
            
            # Add the image to the sheet
            plot_sheet.add_image(img, 'A1')
            
            # Add title and description
            plot_sheet['A30'] = 'Roofline Analysis Plot'
            plot_sheet['A31'] = f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}'
            
    return output_file