###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# adapted from TritonBlas's config generator
# https://github.com/ROCm/tritonBLAS/blob/main/include/tritonblas/origami.py
# an exhaustive list can also be pulled from hipblaslt https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Origami

import itertools
import origami
from math import ceil, gcd
import warnings


class OrigamiHelper:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        b: int,
        a_dtype: origami.data_type_t,
        b_dtype: origami.data_type_t,
        out_dtype: origami.data_type_t,
        hardware: origami.hardware_t,
        mx_block_size=0,
        streamk=False,
    ):
        self._m = m
        self._n = n
        self._k = k
        self._b = b
        self.streamk = streamk
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.out_dtype = out_dtype
        self._hardware = hardware
        self._mx_block_size = mx_block_size

        #####
        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = [origami.data_type_t.Float4, origami.data_type_t.Float6]

        def get_dtype_bits(dtype):
            return origami.datatype_to_bits(dtype)

        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(b_dtype)
        self._out_dtype_bitsize = get_dtype_bits(out_dtype)

        # For matrix instruction latency lookup, use input dtype (not output dtype)
        # because the matrix instruction type is determined by input operand types
        # Example: FP8 inputs with BF16 output still uses FP8 matrix instructions
        # Set MI dtype - use string for MX types, otherwise lookup from dict
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            self.mi_dtype = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
        #####

        # Get hardware info from Origami
        self._N_CU = self._hardware.N_CU

        # Create list of Origami config_t objects from defaults.
        self._block_mn_range = [16, 32, 64, 128, 256]
        self._block_k_range = [16, 32, 64, 128, 256, 512]
        self._kernel_occupancy_range = [1]
        self._configs = self._generate_default_configs()

        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Run Origami solution selection
        self._result = origami.select_config(
            self._problem, self._hardware, self._configs
        )

        # not needed for now
        # if streamk:
        #     self._grid = self._compute_sk_grid()
        # else:
        #     self._grid = self._hardware.N_CU

        # self._xcc_workgroup_mapping, self._workgroup_mapping = (
        #     origami.select_workgroup_mapping(
        #         self._problem, self._hardware, self._result.config, self._grid
        #     )
        # )

    @property
    def block_m(self):
        return self._result.config.mt.m

    @property
    def block_n(self):
        return self._result.config.mt.n

    @property
    def block_k(self):
        return self._result.config.mt.k

    @property
    def group_m(self):
        return self._workgroup_mapping

    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping

    @property
    def waves_per_eu(self):
        return self._result.config.occupancy

    @property
    def even_k(self):
        return gcd(self._k, self.block_k) == self.block_k

    @property
    def sk_grid(self):
        return self._grid

    @property
    def result(self):
        return self._result

    def _compute_sk_grid(self):
        # Grid model constants for StreamK
        split_factors = [8, 6, 4, 3, 2, 1]
        tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
        max_workspace = 128 * 1024 * 1024

        M, N, K = self._m, self._n, self._k
        BLK_M, BLK_N, BLK_K = self.block_m, self.block_n, self.block_k
        cu_count = self._hardware.N_CU

        # Fallback if no better fractional split is found
        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        # More tiles than CUs: try fractional splits to distribute work
        if tiles > cu_count:
            virt_cu_count = cu_count
            # if size_mapping.CUOccupancy > 1:
            # virt_cu_count *= size_mapping.CUOccupancy

            # Try these fractional denominators in order
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (
                    tiles % frac_grid != 0
                    and self._partial_tile_size(frac_grid) > max_workspace
                ):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif tiles < cu_count:
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            last_wave_occupancy = last_wave_remainder / cu_count

            # Really bad last wave, which would have originally been compensated for
            # by changing tile size, but triton tile sizes are limited
            if (
                last_wave_remainder < 128
                and last_wave_remainder > 0
                and cu_count in [304, 80, 64]
            ):  # gfx942
                sk_grid = 256 if cu_count == 304 else 64
        return sk_grid

    def _partial_tile_size(self, sk_grid: int) -> int:
        """
        Python equivalent of ContractionSolution::partialTileSize.

        workspaceSizePerElemC = (element_size_out bits) / 8 → bytes per output element

        tileSize = BLK_M * BLK_N * workspaceSizePerElemC
        return tileSize * sk_grid
        """
        # get the macro-tile dims you already compute
        BLK_M, BLK_N = self.block_m, self.block_n

        # bytes per C element
        bytes_per_elem = self._out_dtype_bitsize // 8

        # size of one partial tile per WG
        tile_size = BLK_M * BLK_N * bytes_per_elem

        # scale by the number of partial‑tiles per WG
        return tile_size * sk_grid

    def _generate_default_configs(self):
        config_list = []

        mi = self._infer_matrix_instruction_dimensions()

        for blk_m, blk_n, blk_k, occupancy in itertools.product(
            self._block_mn_range,
            self._block_mn_range,
            self._block_k_range,
            self._kernel_occupancy_range,
        ):
            # Create special dim3_t object for BLK_* sizes
            mt = origami.dim3_t(blk_m, blk_n, blk_k)

            # Create and set new config_t values
            new_config = origami.config_t()
            new_config.mt = mt
            new_config.mi = mi
            new_config.occupancy = occupancy
            if self.streamk:
                new_config.grid_selection = origami.grid_selection_t.k_split_aware
            else:
                new_config.grid_selection = origami.grid_selection_t.data_parallel
            config_list.append(new_config)

        return config_list

    def _make_problem(self) -> origami.problem_t:
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size = size
        problem.batch = self._b
        problem.a_transpose = origami.transpose_t.N
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype = self.a_dtype
        problem.b_dtype = self.b_dtype
        problem.c_dtype = self.out_dtype
        problem.d_dtype = self.out_dtype
        problem.mi_dtype = self.mi_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size

        return problem

    def _infer_matrix_instruction_dimensions(self):
        """
        Infers the matrix instruction dimensions based on the hardware configuration
        and the sizes of the input data types.  The input dtype sizes are retrieved
        from local object variables.

        Returns:
            origami.dim3_t: An Origami dimension trio containing the matrixinstruction
                dimensions [M, N, K].

        Raises:
            ValueError: If the hardware architecture is unsupported or if the data type
                sizes are not compatible with the detected hardware.
        """
        largest_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)

        mi_dim = None
        # gfx950
        if self._hardware.N_CU == 256:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6F8
            if largest_bitsize <= 8:
                if self._k % 256 == 0:
                    self._block_k_range = self._block_k_range + [256]
                else:
                    self._block_k_range = self._block_k_range + [128]
                self._block_mn_range = [32, 64, 128, 256]
                mi_dim = origami.dim3_t(16, 16, 128)
        # gfx942 (304 CUs full, 80 CUs partitioned, 64 CUs)
        is_gfx942 = self._hardware.N_CU in [304, 80, 64]
        if is_gfx942:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on gfx942
            if largest_bitsize < 8:
                raise ValueError("gfx942 doesn't support F4/F6")
        if self._hardware.N_CU == 228:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            # F8
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            # F4F6 -> Unsupported on MI300A
            if largest_bitsize < 8:
                raise ValueError("MI300A doesn't support F4/F6")
        # gfx90a
        if self._hardware.N_CU == 104:
            # FP32
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            # FP16/BF16
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                raise ValueError("MI200 doesn't support F8")
            if largest_bitsize < 8:
                raise ValueError("MI200 doesn't support F4/F6")
        # Architecture Detected is not valid
        if mi_dim == None:
            raise ValueError(
                f"No Valid Matrix Instruction integrated for {self._a_dtype_bitsize}-bit or {self._b_dtype_bitsize}-bit datatypes"
            )

        return mi_dim

    def get_simulation_time(self) -> float:
        """
        Returns the simulation time in microseconds.
        """
        return self._result.latency / (self._hardware.compute_clock_ghz * 1e3)

    @staticmethod
    def get_hardware(arch: dict) -> origami.hardware_t:
        # find architecture name, look up
        # need to map from GPU name ("mi300x") to internal name ("gfx42a")
        arch_name = arch.get("name", None)
        if arch_name is None:
            warnings.warn("No architecture name provided; Assuming MI300x")
            arch_name = "mi300x"
        freq_mhz = arch.get("freq_mhz", None)
        if freq_mhz is None:
            warnings.warn("No frequency provided; Assuming 2200 MHz")
            freq_mhz = 2200
        arch_name = arch_name.lower()
        if arch_name.endswith("x"):
            arch_name = arch_name[:-1]
        name_to_arch = {
            "mi300": "gfx942",
            "mi325": "gfx942",
            "mi350": "gfx950",
            "mi355": "gfx950",
        }
        arch_name = name_to_arch.get(arch_name, arch_name)

        # The origami python bindings only support the 3xx series
        arch_to_props = {
            # Architecture strings
            "gfx942": {
                "arch": origami.architecture_t.gfx942,
                "N_CU": 304,
                "lds_capacity": 64 * 1024,  # 64 KB
                "L2_capacity": 4 * 1024 * 1024,  # 4 MB
            },
            "gfx950": {
                "arch": origami.architecture_t.gfx950,
                "N_CU": 256,
                "lds_capacity": 64 * 1024,  # 64 KB
                "L2_capacity": 4 * 1024 * 1024,  # 4 MB
            },
        }

        props = arch_to_props[arch_name]  # throw an error for unknown devices
        return origami.get_hardware_for_arch(
            arch=props["arch"],
            N_CU=props["N_CU"],
            lds_capacity=props["lds_capacity"],
            L2_capacity=props["L2_capacity"],
            compute_clock_khz=freq_mhz * 1000,
        )
