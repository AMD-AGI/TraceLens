import argparse
import os
import sys

from TraceLens.PerfModel.perf_model import GEMM, SDPA, gemmologist_dtype_map

def main():
    parser = argparse.ArgumentParser(description="Simulate GEMM or SDPA operation time using perf_model")
    parser.add_argument("--op", choices=["gemm", "sdpa"], required=True, help="Operation type: gemm or sdpa")
    parser.add_argument("--M", type=int, help="Matrix dimension M (GEMM)")
    parser.add_argument("--N", type=int, help="Matrix dimension N (GEMM)")
    parser.add_argument("--K", type=int, help="Matrix dimension K (GEMM)")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype", type=str, required=True, choices=['fp32', 'bf16', 'fp16', 'int8'], help="Data type (e.g., fp32, bf16, fp16, int8)")
    parser.add_argument("--arch_name", type=str, required=True, help="Architecture name")
    parser.add_argument("--freq_mhz", type=int, required=True, help="Frequency in MHz")
    parser.add_argument("--mem_bw_gbps", type=int, required=True, help="Memory bandwidth in GBps")
    parser.add_argument("--l1_bw_gbps", type=int, required=True, help="L1 bandwidth in GBps")
    parser.add_argument("--num_cus", type=int, required=True, help="Number of compute units")
    parser.add_argument("--gemm_units_per_cu", type=int, default=4, help="Number of gemm units per CU")
    parser.add_argument("--backward", action="store_true", help="Simulate backward pass")
    parser.add_argument("--python_path", type=str, default=None, help="Path to Python executable")

    # SDPA-specific
    parser.add_argument("--H_Q", type=int, help="Number of heads for Q (SDPA)")
    parser.add_argument("--N_Q", type=int, help="Sequence length for Q (SDPA)")
    parser.add_argument("--H_KV", type=int, help="Number of heads for K/V (SDPA)")
    parser.add_argument("--N_KV", type=int, help="Sequence length for K/V (SDPA)")
    parser.add_argument("--d_h", type=int, help="Head dimension (SDPA)")

    args = parser.parse_args()
    dict_dtype_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1}
    bytes_per_element = dict_dtype_bytes[args.dtype]

    arch = {
        "name": args.arch_name,
        "freq_mhz": args.freq_mhz,
        "mem_bw_gbps": args.mem_bw_gbps,
        "l1_bw_gbps": args.l1_bw_gbps,
        "gemm_units_per_cu": args.gemm_units_per_cu,
        "num_cus": args.num_cus
    }

    if args.op == "gemm":
        #print("Calling GEMM.get_simulation_time_func...")
        # check if GEMMOLOGIST_PATH is set, otherwise give error message
        if (not os.environ.get('GEMMOLOGIST_PATH')) or \
                (not os.path.exists(os.environ.get('GEMMOLOGIST_PATH'))):
            raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
        time, cmd = GEMM.get_simulation_time_func(
            arch=arch, M=args.M, N=args.N, K=args.K, B=args.B,
            dtype=args.dtype, python_path=args.python_path
        )
        print(f"GEMM simulation time (us): {time}")
        flops = GEMM.flops_func(args.M, args.N, args.K, None)
        tflops_per_gpu_per_s = (flops / 1e12) / (time / 1e6) if time > 0 else float('nan')
        print(f"GEMM TFLOPS/GPU/s: {tflops_per_gpu_per_s}")
        print(f"Command used: {cmd}")

    elif args.op == "sdpa":
        if None in {args.H_Q, args.N_Q, args.N_KV, args.d_h}:
            raise ValueError("For SDPA, --H_Q, --N_Q, --N_KV, and --d_h are required.")
        # check if GEMMOLOGIST_PATH is set, otherwise give error message
        if (not os.environ.get('GEMMOLOGIST_PATH')) or \
                (not os.path.exists(os.environ.get('GEMMOLOGIST_PATH'))):
            raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
        dtype_A_B = gemmologist_dtype_map(args.dtype)
        if args.backward:
            bytes = SDPA.bytes_bwd_func(args.B, args.N_Q, args.H_Q, args.N_KV, args.H_KV, args.d_h, None, bytes_per_element)
            flops = SDPA.flops_bwd_func(args.B, args.N_Q, args.H_Q, args.N_KV, args.H_KV, args.d_h, None, flash_impl=True)
            #print("Calling SDPA.get_simulation_time_bwd_func...")
            time = SDPA.get_simulation_time_bwd_func(
                arch=arch, dtype=args.dtype, python_path=args.python_path,
                dtype_A_B=dtype_A_B, bytes=bytes,
                B=args.B, H_Q=args.H_Q, N_Q=args.N_Q, N_KV=args.N_KV, d_h=args.d_h
            )
            print(f"SDPA Backward simulated time (us): {time}")
        else:
            bytes = SDPA.bytes_func(args.B, args.N_Q, args.H_Q, args.N_KV, args.H_KV, args.d_h, None, bytes_per_element)
            flops = SDPA.flops_func(args.B, args.N_Q, args.H_Q, args.N_KV, args.H_KV, args.d_h, None)
            #print("Calling SDPA.get_simulation_time_func...")
            time = SDPA.get_simulation_time_func(
                arch=arch, dtype=args.dtype, python_path=args.python_path,
                dtype_A_B=dtype_A_B, bytes=bytes,
                B=args.B, H_Q=args.H_Q, N_Q=args.N_Q, N_KV=args.N_KV, d_h=args.d_h
            )
            print(f"SDPA Forward simulated time (us): {time}")
        tflops_per_gpu_per_s = (flops / 1e12) / (time / 1e6) if time > 0 else float('nan')
        print(f"SDPA TFLOPS/GPU/s: {tflops_per_gpu_per_s}")

if __name__ == "__main__":
    main()
