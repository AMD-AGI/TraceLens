
# keywords for splitting jax events
GemmKeys = ["Cijk", "gemm", "nvjet", "cublasLt"]
FABwdKeys = ["FmhaBwd"]
FAFwdKeys = ["FmhaFwd"]
FAV3Keys = ["kernel_func"] # find a more precise way to do this
ConvKeys = ["FillBuffer"]
TEKeys = ["transformer_engine"]
ClassCategories = {
    "GEMM": GemmKeys,
    "FA BWD": FABwdKeys,
    "FA FWD": FAFwdKeys,
    "FA V3": FAV3Keys,
    "Conv": ConvKeys,
    "TE": TEKeys,
}
UncategorizedEventKey = "Uncategorized Events"

def categorize_jax_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', ... or 'other'.
    """

    debug = False
    if any(f in row['name'] for f in GemmKeys):
        return 'GEMM'
    return 'other'