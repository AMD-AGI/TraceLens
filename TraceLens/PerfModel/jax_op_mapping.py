import os, sys
from collections import defaultdict

from . import perf_model
from ..util import TraceEventUtils
from ..TreePerf.jax_analyses import JaxAnalyses

"""
Reuse modules and variables from TreePerf/jax_analyses.py to enable perf analysis with Jax TraceToTree.
"""

dict_jax_category2class = {
    "GEMM": JaxAnalyses.JaxGemm
    #"Conv": JaxConv
    #"TE": JaxTE
    #"FA V3": JaxFaV3
}

def categorize_jax_op(event):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', ... or 'other'.
    """
    name = event[TraceEventUtils.TraceKeys.Name] # event['name']
    cats = [cat for cat, keys in TraceEventUtils.JaxOpKeys.ClassCategories.items() if any(k in name for k in keys)] 
    if len(cats)==1:
        return cats[0]
    elif len(cats)==0:
        return TraceEventUtils.JaxOpKeys.UncategorizedEventKey
    else:
        print('Multiple cats found for event.', cats)
        return ','.join(cats)
    
    
# kernel event: Input Dims Input type Input Strides Concrete Inputs
