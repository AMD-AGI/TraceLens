#!/usr/bin/env python3
import typer
from pathlib import Path
from typing import Optional

from cli import read_toml_config, display_summary
from xlsx import read_xlsx, add_analysis_columns, export_to_xlsx
from roofline import calculate_ridge_points, calculate_rooflines, plot_roofline

app = typer.Typer(help="Roofline Analysis Tool")

@app.command()
def analyze(
    excel_file: Path = typer.Argument(..., help="Path to the Excel file containing kernel data"),
    config_file: Path = typer.Argument(..., help="Path to the TOML config file"),
    plot_output: Optional[Path] = typer.Option(None, help="Path to save a separate copy of the roofline plot"),
    skip_plot: bool = typer.Option(False, help="Skip generating the roofline plot")
):
    """
    Analyze kernel performance data using the roofline model.
    """
    try:
        # Read configuration
        typer.echo(f"Reading configuration from {config_file}")
        config = read_toml_config(str(config_file))
        
        # Read Excel data
        typer.echo(f"Reading data from {excel_file}")
        df = read_xlsx(str(excel_file), config)
        
        # Save a copy of the original data
        df_orig = df.copy()
        
        # Calculate ridge points and rooflines
        typer.echo("Calculating roofline model parameters")
        ai_ridge_mtf, ai_ridge_maf = calculate_ridge_points(config)
        rooflines = calculate_rooflines(config)
        
        # Add analysis columns
        typer.echo("Analyzing kernel performance")
        df_result = add_analysis_columns(df, config, ai_ridge_mtf, ai_ridge_maf)
        
        # Generate roofline plot
        plot_fig = None
        if not skip_plot:
            typer.echo("Generating roofline plot")
            plot_fig = plot_roofline(df_result, rooflines, config, ai_ridge_mtf, ai_ridge_maf, 
                                     str(plot_output) if plot_output else None)
        
        # Export results
        typer.echo("Exporting results to Excel (including plot and original data)")
        output_file = export_to_xlsx(df_result, df_orig, config, ai_ridge_mtf, ai_ridge_maf, plot_fig)
        
        # Display summary
        display_summary(df_orig, df_result, output_file, config, ai_ridge_mtf, ai_ridge_maf)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()