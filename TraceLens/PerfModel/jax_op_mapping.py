from . import perf_model

jax_op_to_perf_model_class_map = {
    "jax_gemm": perf_model.jax_gemm, # JaxAnalyses.JaxGemm, # Legacy
    "jax_te_fused_attn_forward": perf_model.jax_te_fused_attn_forward,
    "jax_te_fused_attn_backward": perf_model.jax_te_fused_attn_backward,
    "jax_conv": perf_model.jax_conv,
}

