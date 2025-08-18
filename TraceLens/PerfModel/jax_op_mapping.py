# keywords for splitting jax events
GemmKeys = ["Cijk", "gemm", "nvjet", "cublasLt"]
FABwdKeys = ["FmhaBwd", ]
FAFwdKeys = ["FmhaFwd", ]
FAV3Keys = ["kernel_func", ] 
ConvKeys = ["FillBuffer", ]
TEKeys = ["transformer_engine", ]
ClassCategories = {
    "GEMM": GemmKeys,
    "FA BWD": FABwdKeys,
    "FA FWD": FAFwdKeys,
    "FA V3": FAV3Keys,
    "Conv": ConvKeys,
    "TE": TEKeys,
}
UncategorizedEventKey = "Uncategorized Events"

dict_cat2names_jax = None

def categorize_jax_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', ... or 'other'.
    """

    debug = False
    for category, filters in ClassCategories.items():
        if any(f in row['name'] for f in filters):
            return category
        return 'other'
    
# kernel event: Input Dims Input type Input Strides Concrete Inputs