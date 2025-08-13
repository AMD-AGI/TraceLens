from TraceLens.TreePerf import JaxAnalyses

def categorize_jax_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', ... or 'other'.
    """

    debug = False
    if any(f in row['name'] for f in JaxAnalyses.GemmKeys):
        return 'GEMM'
    return 'other'