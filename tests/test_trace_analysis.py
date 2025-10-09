from TraceLens.TreePerf import JaxTreePerfAnalyzer

def test_gpu_pids():
    jax_conv_minimal='./data/jax_conv_minimal/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb'
    perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=jax_conv_minimal)
    gpu_pids = perf_analyzer.gpu_event_analyser.gpu_pids
    assert len(gpu_pids) > 0

