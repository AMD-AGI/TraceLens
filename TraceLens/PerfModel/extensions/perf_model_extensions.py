###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for pseudo-op extensions.
"""

from TraceLens.PerfModel.utils import torch_dtype_map


# ==============================================================================
# MoE Performance Models
# ==============================================================================

class FusedMoE:
    """
    Base class for Fused MoE operations.

    Fused MoE operations combine the entire MoE computation (up/gate projection,
    activation, and down projection) into a single kernel launch.
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path

    @staticmethod
    def flops_func(num_tokens, hidden_dim, inter_dim, topk, gated):
        """
        Calculate FLOPs for MoE forward pass.
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
            gated (bool): Whether gated activation is used (e.g., SwiGLU)
        
        Returns:
            int: Total FLOPs for the MoE operation
        """
        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        # FC1: M×K @ K×N for each of topk experts (×2 if gated)
        fc1_flops = 2 * M * K * N * topk * (2 if gated else 1)
        
        # Activation FLOPs (ignored, activation-dependent?)
        activation_flops = 0
        
        # FC2: M×N @ N×K for each of topk experts
        fc2_flops = 2 * M * K * N * topk
        
        # Aggregation: weighted sum of expert outputs
        # For each output element: multiply by weight (topk ops) + sum (topk-1 ops)
        aggregation_flops = M * K * (2 * topk - 1)
        
        total_flops = fc1_flops + activation_flops + fc2_flops + aggregation_flops
        
        return total_flops
    
    @staticmethod
    def bytes_func(num_tokens, hidden_dim, inter_dim, topk, gated, 
                   input_bpe, weight_bpe, output_bpe):
        """
        Calculate bytes moved for fused MoE forward pass.
        
        For fused MoE, only count:
        - Input: M×K
        - FC1 weights: topk × N×K (×2 if gated)
        - FC2 weights: topk × N×K
        - Output: M×K
        
        Ignores intermediate activations and scales (fused operation).
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
            gated (bool): Whether gated activation is used
            input_bpe (int): Bytes per element for input
            weight_bpe (int): Bytes per element for weights
            output_bpe (int): Bytes per element for output
        
        Returns:
            int: Total bytes moved
        """
        if None in {input_bpe, weight_bpe, output_bpe}:
            return None
        
        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        input_bytes = M * K * input_bpe
        fc1_weight_bytes = topk * N * K * weight_bpe * (2 if gated else 1)
        fc2_weight_bytes = topk * N * K * weight_bpe
        output_bytes = M * K * output_bpe
        
        total_bytes = input_bytes + fc1_weight_bytes + fc2_weight_bytes + output_bytes
        
        return total_bytes


class moe_aiter_fused_1stage(FusedMoE):
    """
    Performance model for only AITER-based fused MoE operation. Handles AITER fused_moe_1stage launches.

    TO DO: Expand support for other AITER MoE kernels.
    """
    
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        """
        Extract MoE dimensions and data types from event args.
        
        Expected Input Dims format (from vllm::rocm_aiter_fused_moe):
        [[tokens, hidden_dim], [experts, inter_dim×(gated+1), hidden_dim], 
         [experts, hidden_dim, inter_dim], [tokens, topk], ...]
        
        Expected Input type format:
        [dtype_input, dtype_w1, dtype_w2, dtype_topk_weights, ...]
        """

        args = event.get('args', {})
        
        kernel_input_shape = args['Input Dims']
        input_shape = kernel_input_shape[0]
        w1_shape = kernel_input_shape[1]
        w2_shape = kernel_input_shape[2]
        topk_weights_shape = kernel_input_shape[3]
        
        num_tokens = input_shape[0]
        hidden_dim = input_shape[1]
        inter_dim = w2_shape[2]
        num_experts = w1_shape[0]
        topk = topk_weights_shape[1]
        
        # Check if MoE is using gated activation (SwiGLU)
        gated = (w1_shape[1] == 2 * inter_dim)
        
        input_dtype = args['Input type'][0]
        weight_dtype = args['Input type'][1]
        
        return {
            'num_tokens': num_tokens,
            'hidden_dim': hidden_dim,
            'inter_dim': inter_dim,
            'num_experts': num_experts,
            'topk': topk,
            'gated': gated,
            'input_dtype': input_dtype,
            'weight_dtype': weight_dtype,
        }
    
    def flops(self):
        """Calculate FLOPs using the static flops_func."""

        return self.flops_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk'],
            self.param_details['gated']
        )

    
    def bytes(self):
        """Calculate bytes moved using the static bytes_func."""

        # Map dtype strings to byte sizes
        dtype_to_bytes = {
            'Float8_e4m3fn': 1,
            'Float8_e4m3fnuz': 1,
            'Float8_e5m2': 1,
            'Float8_e5m2fnuz': 1,
            'BFloat16': 2,
            'Float16': 2,
            'Half': 2,
            'Float32': 4,
            'Float': 4,
            'c10::BFloat16': 2,
            'c10::Float8_e4m3fn': 1,
            'c10::Float8_e4m3fnuz': 1,
            'c10::Half': 2,
            'c10::Float': 4,
        }
        
        input_bpe = dtype_to_bytes.get(self.param_details['input_dtype'], 2)  # Default to 2
        weight_bpe = dtype_to_bytes.get(self.param_details['weight_dtype'], 1)  # Default to 1 (FP8)
        output_bpe = input_bpe  # Output typically same as input
        
        return self.bytes_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk'],
            self.param_details['gated'],
            input_bpe,
            weight_bpe,
            output_bpe
        )
    
    def flops_bwd(self):
        """Backward pass FLOPs (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for fused MoE is not defined.")
    
    def bytes_bwd(self):
        """Backward pass bytes (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for fused MoE is not defined.")
    
    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        dtype = self.param_details.get("input_dtype")
        return torch_dtype_map(dtype) if dtype else None
    
    def get_maf_type(self):
        """Return the MAF type for this operation (matrix for MoE)."""
        return "matrix"


class UnfusedMoE_Up:
    """
    Base class for Unfused MoE up projection operations.
    
    Handles the first stage of unfused MoE which performs:
    - Up projection: [tokens, hidden_dim] → [tokens, inter_dim]
    - Optionally gated (e.g., SwiGLU): both up and gate projections
        """

    @staticmethod
    def flops_func(num_tokens, hidden_dim, inter_dim, topk, gated):
        """
        Calculate FLOPs for unfused MoE up projection.
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
            gated (bool): Whether gated activation is used (e.g., SwiGLU)
        
        Returns:
            int: Total FLOPs for up projection stage
        """

        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        # Up projection: M×K @ K×N for each of topk experts. If gated (e.g., SwiGLU), multiply by 2 for up+gate projections
        gating_factor = 2 if gated else 1
        up_flops = 2 * M * K * N * topk * gating_factor
        
        return up_flops

    @staticmethod
    def bytes_func(num_tokens, hidden_dim, inter_dim, topk, gated,
                   input_bpe, weight_bpe, output_bpe):
        """
        Calculate bytes moved for unfused MoE up projection.
        
        For unfused up projection:
        - Read: M×K (input) + topk×gating_factor×K×N (weights)
        - Write: M×N (intermediate output)
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
            gated (bool): Whether gated activation is used
            input_bpe (int): Bytes per element for input
            weight_bpe (int): Bytes per element for weights
            output_bpe (int): Bytes per element for output
        
        Returns:
            int: Total bytes moved
        """

        if None in {input_bpe, weight_bpe, output_bpe}:
            return None
        
        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        gating_factor = 2 if gated else 1
        
        input_bytes = M * K * input_bpe
        weight_bytes = topk * gating_factor * K * N * weight_bpe
        output_bytes = M * N * output_bpe
        
        total_bytes = input_bytes + weight_bytes + output_bytes
        
        return total_bytes


class UnfusedMoE_Down:
    """
    Base class for Unfused MoE down projection operations.
    
    Handles the second stage of unfused MoE which performs:
    - Down projection: [tokens, inter_dim] → [tokens, hidden_dim]
    
    This base class only provides static calculation functions.
    Child classes implement get_param_details() to extract parameters from events.
    """

    @staticmethod
    def flops_func(num_tokens, hidden_dim, inter_dim, topk):
        """
        Calculate FLOPs for unfused MoE down projection.
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
        
        Returns:
            int: Total FLOPs for down projection stage
        """
        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        # Down projection: M×N @ N×K for each of topk experts
        down_flops = 2 * M * N * K * topk
        
        return down_flops

    @staticmethod
    def bytes_func(num_tokens, hidden_dim, inter_dim, topk,
                   input_bpe, weight_bpe, output_bpe):
        """
        Calculate bytes moved for unfused MoE down projection.
        
        For unfused down projection:
        - Read: M×N (intermediate input) + topk×N×K (weights)
        - Write: M×K (output)
        
        Args:
            num_tokens (int): Number of input tokens (M)
            hidden_dim (int): Hidden dimension size (K)
            inter_dim (int): Intermediate dimension size (N)
            topk (int): Number of experts per token
            input_bpe (int): Bytes per element for input
            weight_bpe (int): Bytes per element for weights
            output_bpe (int): Bytes per element for output
        
        Returns:
            int: Total bytes moved
        """
        if None in {input_bpe, weight_bpe, output_bpe}:
            return None
        
        M = num_tokens
        K = hidden_dim
        N = inter_dim
        
        input_bytes = M * N * input_bpe
        weight_bytes = topk * N * K * weight_bpe
        output_bytes = M * K * output_bpe
        
        total_bytes = input_bytes + weight_bytes + output_bytes
        
        return total_bytes


class moe_triton_unfused_up(UnfusedMoE_Up):
    """
    Performance model for Triton-based unfused MoE up projection stage (Applicable to GPTOSS)
    
    Handles the first stage of unfused MoE which performs:
    - Up projection: [tokens, hidden_dim] → [tokens, inter_dim]
    - Optionally gated (e.g., SwiGLU): both up and gate projections

    LIMITATION: Perf. model assumes that the inter_dim is equal to the hidden_dim. (Not available in Trace)
    """
    
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        """
        Extract MoE up projection parameters from event args.
        
        Expected args structure (from moe_unfused_pseudo_ops.py):
        - Input Dims: [[tokens, hidden_dim], [tokens, num_experts], ...]
        - MoE GEMM type: 'up'
        - MoE GEMM gated: True/False
        - MoE topk: Number of active experts per token
        
        Raises:
            KeyError: If required args keys are missing
            ValueError: If extracted values are invalid
        """
        args = event.get('args', {})
        
        # Extract Input Dims
        input_dims = args['Input Dims']
        if len(input_dims) < 2:
            raise ValueError(
                f"Expected at least 2 Input Dims for unfused MoE, got {len(input_dims)}"
            )
        
        input_shape = input_dims[0]  # [tokens, hidden_dim]
        router_shape = input_dims[1]  # [tokens, num_experts]
        
        num_tokens = input_shape[0]
        hidden_dim = input_shape[1]
        num_experts = router_shape[1]
        
        # Extract topk (REQUIRED)
        if 'MoE topk' not in args:
            raise KeyError(f"'MoE topk' not found in event args")
        topk = args['MoE topk']
        
        # Extract gated flag (REQUIRED)
        if 'MoE GEMM gated' not in args:
            raise KeyError(f"'MoE GEMM gated' not found in event args")
        gated = args['MoE GEMM gated']
        
        # LIMITATION: inter_dim is not present in the trace (GPTOSS default used)
        inter_dim = hidden_dim
        
        # Detect weight dtype from kernel name (may be quantized)
        weight_dtype_actual = None
        if "kernel_details" in event and event["kernel_details"]:
            kernel_name = event["kernel_details"][0].get("name", "")
            if "mxfp4" in kernel_name.lower() or "fp4" in kernel_name.lower():
                weight_dtype_actual = "FP4"
            elif "fp8" in kernel_name.lower() or "e4m3" in kernel_name.lower():
                weight_dtype_actual = "FP8"
        else:
            raise ValueError(f"Kernel details not found in event")
        
        # Extract data types
        input_types = args.get('Input type', [])
        if len(input_types) < 2:
            raise ValueError(f"Expected at least 2 Input types, got {len(input_types)}")
        
        input_dtype = input_types[0]
        
        return {
            'num_tokens': num_tokens,
            'hidden_dim': hidden_dim,
            'inter_dim': inter_dim,
            'num_experts': num_experts,
            'topk': topk,
            'gated': gated,
            'input_dtype': input_dtype,
            'weight_dtype': weight_dtype_actual,
        }
    
    def flops(self):
        """Calculate FLOPs for up projection."""
        return self.flops_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk'],
            self.param_details['gated']
        )
    
    def bytes(self):
        """Calculate bytes moved for up projection."""
        # Map dtype strings to byte sizes
        dtype_to_bytes = {
            'Float8_e4m3fn': 1,
            'Float8_e4m3fnuz': 1,
            'Float8_e5m2': 1,
            'Float8_e5m2fnuz': 1,
            'FP8': 1,
            'FP4': 0.5,
            'BFloat16': 2,
            'Float16': 2,
            'Half': 2,
            'Float32': 4,
            'Float': 4,
            'c10::BFloat16': 2,
            'c10::Float8_e4m3fn': 1,
            'c10::Float8_e4m3fnuz': 1,
            'c10::Half': 2,
            'c10::Float': 4,
        }
        
        input_dtype = self.param_details['input_dtype']
        weight_dtype = self.param_details['weight_dtype']
        
        if input_dtype not in dtype_to_bytes:
            raise ValueError(f"Unknown input dtype '{input_dtype}'")
        if weight_dtype not in dtype_to_bytes:
            raise ValueError(f"Unknown weight dtype '{weight_dtype}'")
        
        input_bpe = dtype_to_bytes[input_dtype]
        weight_bpe = dtype_to_bytes[weight_dtype]
        output_bpe = input_bpe  # Output same dtype as input
        
        return self.bytes_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk'],
            self.param_details['gated'],
            input_bpe,
            weight_bpe,
            output_bpe
        )
    
    def flops_bwd(self):
        """Backward pass FLOPs (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for unfused MoE is not defined.")
    
    def bytes_bwd(self):
        """Backward pass bytes (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for unfused MoE is not defined.")
    
    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        dtype = self.param_details.get("weight_dtype")
        return torch_dtype_map(dtype) if dtype else None
    
    def get_maf_type(self):
        """Return the MAF type for this operation (matrix for MoE)."""
        return "matrix"


class moe_triton_unfused_down(UnfusedMoE_Down):
    """
    Performance model for Triton-based unfused MoE down projection stage (Applicable to GPTOSS)
    
    Handles the second stage of unfused MoE which performs:
    - Down projection: [tokens, inter_dim] → [tokens, hidden_dim]

    LIMITATION: Perf. model assumes that the inter_dim is equal to the hidden_dim. (Not available in Trace)
    """
    
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        """
        Extract MoE down projection parameters from event args.
        
        Same structure as moe_2stage_up but gated is always False for down projection.
        """
        args = event.get('args', {})
        
        # Extract Input Dims
        input_dims = args['Input Dims']
        if len(input_dims) < 2:
            raise ValueError(
                f"Expected at least 2 Input Dims for unfused MoE, got {len(input_dims)}"
            )
        
        input_shape = input_dims[0]  # [tokens, hidden_dim]
        router_shape = input_dims[1]  # [tokens, num_experts]
        
        num_tokens = input_shape[0]
        hidden_dim = input_shape[1]
        num_experts = router_shape[1]
        
        # Extract topk (REQUIRED)
        if 'MoE topk' not in args:
            raise KeyError(f"'MoE topk' not found in event args")
        topk = args['MoE topk']
        
        # Extract gated flag (REQUIRED) - typically False for down projection
        if 'MoE GEMM gated' not in args:
            raise KeyError(f"'MoE GEMM gated' not found in event args")
        gated = args['MoE GEMM gated']
        
        # LIMITATION: inter_dim is not present in the trace (GPTOSS default used)
        inter_dim = hidden_dim
        
        # Detect weight dtype from kernel name
        weight_dtype_actual = None
        if "kernel_details" in event and event["kernel_details"]:
            kernel_name = event["kernel_details"][0].get("name", "")
            if "mxfp4" in kernel_name.lower() or "fp4" in kernel_name.lower():
                weight_dtype_actual = "FP4"
            elif "fp8" in kernel_name.lower() or "e4m3" in kernel_name.lower():
                weight_dtype_actual = "FP8"
        else:
            raise ValueError(f"Kernel details not found in event")
        
        # Extract data types
        input_types = args.get('Input type', [])
        if len(input_types) < 2:
            raise ValueError(f"Expected at least 2 Input types, got {len(input_types)}")
        
        input_dtype = input_types[0]
        
        return {
            'num_tokens': num_tokens,
            'hidden_dim': hidden_dim,
            'inter_dim': inter_dim,
            'num_experts': num_experts,
            'topk': topk,
            'gated': gated,
            'input_dtype': input_dtype,
            'weight_dtype': weight_dtype_actual,
        }
    
    def flops(self):
        """Calculate FLOPs for down projection."""
        return self.flops_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk']
        )
    
    def bytes(self):
        """Calculate bytes moved for down projection."""
        # Map dtype strings to byte sizes
        dtype_to_bytes = {
            'Float8_e4m3fn': 1,
            'Float8_e4m3fnuz': 1,
            'Float8_e5m2': 1,
            'Float8_e5m2fnuz': 1,
            'FP8': 1,
            'FP4': 0.5,
            'BFloat16': 2,
            'Float16': 2,
            'Half': 2,
            'Float32': 4,
            'Float': 4,
            'c10::BFloat16': 2,
            'c10::Float8_e4m3fn': 1,
            'c10::Float8_e4m3fnuz': 1,
            'c10::Half': 2,
            'c10::Float': 4,
        }
        
        input_dtype = self.param_details['input_dtype']
        weight_dtype = self.param_details['weight_dtype']
        
        if input_dtype not in dtype_to_bytes:
            raise ValueError(f"Unknown input dtype '{input_dtype}'")
        if weight_dtype not in dtype_to_bytes:
            raise ValueError(f"Unknown weight dtype '{weight_dtype}'")
        
        input_bpe = dtype_to_bytes[input_dtype]
        weight_bpe = dtype_to_bytes[weight_dtype]
        output_bpe = input_bpe  # Output same dtype as input
        
        return self.bytes_func(
            self.param_details['num_tokens'],
            self.param_details['hidden_dim'],
            self.param_details['inter_dim'],
            self.param_details['topk'],
            input_bpe,
            weight_bpe,
            output_bpe
        )
    
    def flops_bwd(self):
        """Backward pass FLOPs (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for unfused MoE is not defined.")
    
    def bytes_bwd(self):
        """Backward pass bytes (not implemented for inference-only MoE)."""
        raise NotImplementedError("Backward pass for unfused MoE is not defined.")
    
    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        dtype = self.param_details.get("weight_dtype")
        return torch_dtype_map(dtype) if dtype else None
    
    def get_maf_type(self):
        """Return the MAF type for this operation (matrix for MoE)."""
        return "matrix"
