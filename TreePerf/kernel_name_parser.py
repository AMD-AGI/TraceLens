import re

def gemm_name_parser(kernel_name):
    """
    Parse the kernel name to identify GEMM op details.
    Args:
        kernel_name (str): The name of the kernel.    
    Returns:
        dict: A dictionary containing the GEMM operation details.
    """        
    if is_rocm_gemm(kernel_name):
        return parse_rocm_gemm(kernel_name)
    # NVIDIA trace: check if any of the known NVIDIA markers exist in the kernel name
    elif is_cuda_gemm(kernel_name):
        return parse_cuda_gemm(kernel_name)


def is_rocm_gemm(kernel_name):
    """
    Check if a kernel name matches the more general ROCm GEMM naming pattern.
    Allows an arbitrary prefix before 'Cijk_Alik_Bljk_...' where each of C/A/B 
    is followed by exactly three axis letters.
    Example matches:
      - 'Cijk_Alik_Bljk_...'
      - 'Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_User...'
    """
    pattern = r'^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$'
    return bool(re.match(pattern, kernel_name))

def parse_rocm_gemm(kernel_name):
    
    # 1. Parse the transpose flags from the kernel name
    trans_a, trans_b = None, None
    if "_Ailk_" in kernel_name:
        trans_a = False
    elif "_Alik_" in kernel_name:
        trans_a = True
    if "_Bljk_" in kernel_name:
        trans_b = False
    elif "_Bjlk_" in kernel_name:
        trans_b = True
    
    # Feel free to add more details as needed.
    # https://github.com/ROCm/Tensile/wiki/Kernel-Parameters#kernel-names

    return {'transpose': (trans_a, trans_b)}
    
def is_cuda_gemm(kernel_name):
    """
    Check if a kernel name matches the NVIDIA GEMM naming pattern:
    """
    # Right now, we only check if the kernel name starts with 'nvjet'.
    # This is a temporary solution and will be expanded in the future.
    return kernel_name.startswith('nvjet')

def parse_cuda_gemm(kernel_name):
    transpose_chars = kernel_name.split('_')[-1]
    transpose = transpose_chars[0] == 'T', transpose_chars[1] == 'T'
    return {'transpose': transpose}

