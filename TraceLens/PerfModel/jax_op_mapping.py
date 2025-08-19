import os, sys
from collections import defaultdict

from . import perf_model
from ..util import TraceEventUtils
from ..TreePerf.jax_analyses import JaxAnalyses

"""
Reuse modules and variables from TreePerf/jax_analyses.py to enable perf analysis with Jax TraceToTree.
"""

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

UncategorizedEventKey = JaxAnalyses.UncategorizedEventKey
communication_events_map = JaxAnalyses.communication_events_map

dict_cat_to_perf_model = {
    "GEMM": JaxAnalyses.JaxGemm
}

def categorize_jax_op(event):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', ... or 'other'.
    """

    for category, filters in ClassCategories.items():
        name = event[TraceEventUtils.TraceKeys.Name] # event['name']
        if any(f in name for f in filters):
            return category
        return 'other'
    
# kernel event: Input Dims Input type Input Strides Concrete Inputs