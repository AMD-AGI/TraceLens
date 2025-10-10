import os
import math
from collections import Counter

from TraceLens.TreePerf import JaxTreePerfAnalyzer, TreePerfAnalyzer

print('Working directory:', os.getcwd())
jax_conv_minimal='./data/jax_conv_minimal/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb'
assert os.path.exists(jax_conv_minimal)
perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=jax_conv_minimal)

def profile_jax_conv(path=None):
    if path is None:
        path = "/tmp/jax_trace.xplane.pb"
    # Profile trace
    # /home/guangphu/jax-minimal/jax_conv.py
    return path
   
################## 
# Event statistics
##################

def test_num_tree_events():
    expected_result = 5903
    
    result = len(perf_analyzer.tree.events)
    assert result == expected_result

def test_tree_event_cats():
    expected_result = {'Unknown': 4658, 'cpu_op': 1147, 'memcpy': 53, 'kernel': 25, 'python function': 20}
    
    result = Counter([event['cat'] for event in perf_analyzer.tree.events])
    assert result == expected_result
    
def test_kernel_event_cats():
    expected_result = {'Uncategorized Events/XLA': 15, 'Conv': 10}
    
    result = Counter([event['gpu_kernel_op_cat'] for event in perf_analyzer.tree.events if event['cat'] == 'kernel'])
    assert result == expected_result
    
################
# GPU statistics
################

def test_gpu_pids():
    expected_result = set([1, 8])
    
    result = set(perf_analyzer.gpu_event_analyser.gpu_pids)
    assert result == expected_result
    
def test_gpu_timeline():
    busy_time = perf_analyzer.get_df_gpu_timeline(gpu_pid=1).set_index('type')['time ms']['busy_time']
    assert math.isclose(0.889028, busy_time, rel_tol=1e-5)
    
    busy_time = perf_analyzer.get_df_gpu_timeline(gpu_pid=8).set_index('type')['time ms']['busy_time']
    assert math.isclose(3.586493, busy_time, rel_tol=1e-5)


##################### 
# Performance Metrics
#####################

def test_tree():
    expected_result = None
    
    result = None
    assert result == expected_result


    
