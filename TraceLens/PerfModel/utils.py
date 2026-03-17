###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utils. for perf. model.
"""


def name2bpe(name):
    """
    This function maps a data type name to the number of bytes per element.
    Args:
        name (str): The name of the data type.
    Returns:
        int: The number of bytes per element.
    """
    dict_bpe2dtype = {
        8: ["double", "long int"],
        4: ["float", "scalar"],
        2: ["c10::half", "c10::bfloat16"],
        1: [
            "c10::float8_e4m3fnuz",
            "c10::float8_e4m3fn",
            "c10::float8_e5m2",
            "unsigned char",
            "signed char",
            "fp8",
        ],
    }
    dict_dtype2bpe = {
        dtype: bpe for bpe, dtypes in dict_bpe2dtype.items() for dtype in dtypes
    }
    return dict_dtype2bpe.get(name.lower(), None)


def simulation_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a simulation data type.
    Args:
        dtype (str): The name of the pytorch data type.
    Returns:
        str: The name of the PyTorch data type.
    """
    dict_dtype2simulation = {
        "fp32": "float",
        "fp64": "double",
        "fp16": "c10::half",
        "bf16": "c10::bfloat16",
        "fp8": "c10::float8_e4m3fnuz",
    }
    return dict_dtype2simulation.get(dtype.lower(), None)


def torch_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a simulation data type.
    Args:
        dtype (str): The name of the PyTorch data type.
    Returns:
        str: The name of the simulation data type.
    """
    dict_dtype2simulation = {
        "float": "fp32",
        "double": "fp64",
        "c10::half": "fp16",
        "c10::bfloat16": "bf16",
        "c10::float8_e4m3fnuz": "fp8",
        "unsigned char": "fp8",
        "fp8": "fp8",
    }
    return dict_dtype2simulation.get(dtype.lower(), None)
