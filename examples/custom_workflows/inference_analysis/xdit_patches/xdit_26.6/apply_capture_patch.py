"""Apply the capture trace patch to xDiT's FLUX model.

Patches _compile_model() in flux.py to profile the graph capture phase
(the first forward pass after torch.compile) and save a capture trace
with per-kernel Input Dims / Concrete Inputs for TraceLens shape merging.
"""
import sys

path = "/app/xDiT/xfuser/model_executor/models/runner_models/flux.py"
with open(path) as f:
    code = f.read()

old = '''    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile."""
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead") # Better perf for FLUX.1
        # two steps to warmup the torch compiler
        input_args["num_inference_steps"] = 2
        self._run_timed_pipe(input_args)'''

new = '''    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile."""
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead") # Better perf for FLUX.1
        # two steps to warmup the torch compiler
        input_args["num_inference_steps"] = 2
        # Profile the first forward pass — this is the graph capture phase
        # where torch.compile triggers compilation + autotune + HIP graph
        # recording.  During capture, each kernel dispatches individually,
        # so the profiler records per-kernel cpu_op events with Input Dims,
        # Concrete Inputs, Input type.  This capture trace is saved for
        # TraceLens shape merging.
        import os as _os
        from torch.profiler import profile, record_function, ProfilerActivity
        from xfuser.core.distributed import get_world_group
        from xfuser.core.utils.runner_utils import log
        log("Profiling graph capture phase for shape trace...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as capture_prof:
            with record_function("graph_capture"):
                self._run_timed_pipe(input_args)
        capture_dir = _os.path.join(self.config.output_directory, "capture_traces")
        _os.makedirs(capture_dir, exist_ok=True)
        rank = get_world_group().rank
        capture_path = _os.path.join(capture_dir, f"capture_rank_{rank}.json.gz")
        capture_prof.export_chrome_trace(capture_path)
        log(f"Capture trace saved to {capture_path}", log_from_all_processes=True)'''

if old not in code:
    print("ERROR: patch target not found in base_model.py")
    sys.exit(1)

code = code.replace(old, new)
with open(path, "w") as f:
    f.write(code)
print("Capture trace patch applied successfully")
