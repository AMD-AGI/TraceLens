#!/usr/bin/env python3
"""
TraceLens Runner Module
Handles TraceLens report generation with critical path integration
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "CritPath"))

try:
    from TraceLens.Reporting.generate_perf_report_pytorch_vllm import generate_perf_report_pytorch
    from TraceLens.Reporting.compare_perf_reports_pytorch import generate_compare_perf_reports_pytorch
    TRACELENS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: TraceLens import failed: {e}")
    TRACELENS_AVAILABLE = False

try:
    from link import process_files, prune_spillover_kernels
    from construct_dag import load_trace_and_tree, build_dependency_map
    from dag import DAG
    CRITPATH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: CritPath import failed: {e}")
    CRITPATH_AVAILABLE = False


class TraceLensRunner:
    """Manages TraceLens report generation and critical path analysis"""
    
    def __init__(self, save_intermediates: bool = True, use_critical_path: bool = True, 
                 enable_inference_phase_analysis: bool = False):
        """
        Args:
            save_intermediates: Whether to save intermediate files
            use_critical_path: Whether to use critical path analysis
            enable_inference_phase_analysis: Whether to enable prefill/decode phase analysis
        """
        self.save_intermediates = save_intermediates
        self.use_critical_path = use_critical_path
        self.enable_inference_phase_analysis = enable_inference_phase_analysis
    
    def run_critical_path_analysis(self, kineto: str, et: str, output_dir: Path, 
                                   gpu_name: str) -> Tuple[Path, Path]:
        """
        Run critical path analysis on kineto + ET traces.
        
        Returns:
            Tuple of (linked_trace_path, critical_path_pickle_path)
        """
        if not CRITPATH_AVAILABLE:
            raise ImportError("CritPath modules not available")
        
        print(f"\n{'='*60}")
        print(f"CRITICAL PATH ANALYSIS: {gpu_name}")
        print(f"{'='*60}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Link kineto + ET traces
        print(f"\n1️⃣  Linking traces...")
        linked_trace = output_dir / f"{gpu_name}_linked.json"
        
        try:
            process_files(
                kineto_file=kineto,
                et_file=et,
                output_file=str(linked_trace),
                use_multiprocessing=True
            )
            print(f"✓ Linked trace saved: {linked_trace}")
        except Exception as e:
            print(f"⚠️  Linking failed: {e}")
            print(f"   Falling back to kineto-only trace")
            import shutil
            shutil.copy(kineto, linked_trace)
        
        # Step 2: Build tree and DAG
        print(f"\n2️⃣  Building dependency tree...")
        try:
            trace_data, op_tree = load_trace_and_tree(str(linked_trace))
            if op_tree is None:
                print("⚠️  Could not build operation tree")
                return linked_trace, None
            
            dependency_map = build_dependency_map(trace_data, op_tree)
            print(f"✓ Tree built: {len(op_tree)} nodes")
            
            # Step 3: Find critical path
            print(f"\n3️⃣  Finding critical path...")
            dag = DAG(op_tree, dependency_map)
            dag.find_critical_path()
            
            # Save critical path data
            cp_pickle = output_dir / f"{gpu_name}_critical_path.pkl"
            import pickle
            with open(cp_pickle, 'wb') as f:
                pickle.dump({
                    'dag': dag,
                    'op_tree': op_tree,
                    'dependency_map': dependency_map,
                    'trace_data': trace_data
                }, f)
            
            print(f"✓ Critical path saved: {cp_pickle}")
            print(f"{'='*60}\n")
            
            return linked_trace, cp_pickle
            
        except Exception as e:
            print(f"⚠️  Critical path analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return linked_trace, None
    
    def generate_tracelens_report(self, linked_trace: Path, output_dir: Path,
                                  extension_path: Optional[Path] = None,
                                  gpu_name: str = "GPU") -> Dict:
        """
        Generate TraceLens performance report.
        
        Args:
            linked_trace: Path to linked trace JSON
            output_dir: Output directory for report
            extension_path: Path to critical path extension (if available)
            gpu_name: Name of the GPU
            
        Returns:
            Dictionary of DataFrames from TraceLens report
        """
        if not TRACELENS_AVAILABLE:
            raise ImportError("TraceLens not available")
        
        print(f"\n{'='*60}")
        print(f"GENERATING TRACELENS REPORT: {gpu_name}")
        print(f"{'='*60}\n")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{gpu_name}_tracelens_report.xlsx"
        
        try:
            # Determine if we should use critical path extension
            use_extension = self.use_critical_path and extension_path and extension_path.exists()
            
            # Call TraceLens directly with extension (only if critical path enabled)
            result_dfs = generate_perf_report_pytorch(
                profile_json_path=str(linked_trace),
                output_xlsx_path=str(report_path) if self.save_intermediates else None,
                include_unlinked_kernels=False,
                collective_analysis=True,
                short_kernel_study=False,
                extension_file=str(extension_path) if use_extension else None,
                group_by_parent_module=True
                #inference_phase_analysis=self.enable_inference_phase_analysis
            )
            
            print(f"✓ Report generated: {report_path}")
            
            # Show available sheets
            print(f"  Sheets available: {', '.join(result_dfs.keys())}")
            
            # Show critical path stats if available and enabled
            if self.use_critical_path and 'critical_path_summary' in result_dfs:
                cp_summary = result_dfs['critical_path_summary']
                if not cp_summary.empty and 'critical_path_percentage' in cp_summary.columns:
                    cp_pct = cp_summary['critical_path_percentage'].iloc[0]
                    print(f"  Critical path: {cp_pct:.1f}%")
            
            print(f"{'='*60}\n")
            return result_dfs
            
        except Exception as e:
            print(f"⚠️  TraceLens report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_comparison_report(self, gpu1_report_path: Path, gpu2_report_path: Path,
                                   output_dir: Path, gpu1_name: str, 
                                   gpu2_name: str) -> bool:
        """
        Generate TraceLens comparison report.
        
        Args:
            gpu1_report_path: Path to GPU1 TraceLens report (xlsx)
            gpu2_report_path: Path to GPU2 TraceLens report (xlsx)
            output_dir: Output directory
            gpu1_name: Name of GPU1
            gpu2_name: Name of GPU2
        
        Returns:
            True if successful, False otherwise
        """
        if not TRACELENS_AVAILABLE:
            return False
        
        print(f"\n{'='*60}")
        print(f"GENERATING COMPARISON REPORT")
        print(f"{'='*60}\n")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / f"comparison_{gpu1_name}_vs_{gpu2_name}.xlsx"
        try:
            result_dfs = generate_compare_perf_reports_pytorch(
                reports=[str(gpu1_report_path), str(gpu2_report_path)],
                output=str(comparison_path) if self.save_intermediates else None,
                names=[gpu1_name, gpu2_name]
            )
            print(f"✓ Successfully generated {len(result_dfs)} comparison sheets")
            print(f"✓ Comparison report: {comparison_path}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"⚠️  Comparison report failed: {e}")
            import traceback
            traceback.print_exc()
            return False
