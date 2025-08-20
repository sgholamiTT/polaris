#!/usr/bin/env python3
"""
Polaris Simulation Runner for Generated Workloads

This script automates the process of running Polaris simulations with generated workloads.
It handles configuration setup, workload integration, and result analysis.
"""

import argparse
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import yaml

class PolarisSimulationRunner:
    """Manages Polaris simulation runs for generated workloads"""
    
    def __init__(self, polaris_root: str = "."):
        self.polaris_root = Path(polaris_root)
        self.temp_configs = []
        
    def __del__(self):
        """Cleanup temporary configuration files"""
        for temp_file in self.temp_configs:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def create_combined_workload_config(self, generated_config_path: str, 
                                      base_config_path: str = None) -> str:
        """Create a combined workload configuration including the generated workload"""
        
        # Use default base config if not specified
        if base_config_path is None:
            base_config_path = self.polaris_root / "config" / "all_workloads.yaml"
        
        # Read base configuration
        base_workloads = []
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
                base_workloads = base_config.get('workloads', [])
        
        # Read generated configuration
        with open(generated_config_path, 'r') as f:
            generated_config = yaml.safe_load(f)
            generated_workloads = generated_config.get('workloads', [])
        
        # Combine configurations
        combined_config = {
            'workloads': base_workloads + generated_workloads
        }
        
        # Create temporary combined config file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='polaris_workloads_')
        self.temp_configs.append(temp_path)
        
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(combined_config, f, default_flow_style=False, sort_keys=False)
        
        return temp_path
    
    def run_simulation(self, workload_name: str, workload_instance: str = None,
                      arch_config: str = None, mapping_config: str = None,
                      study_name: str = None, output_dir: str = None,
                      dry_run: bool = False, additional_args: List[str] = None) -> int:
        """Run a Polaris simulation"""
        
        # Default configurations
        if arch_config is None:
            arch_config = self.polaris_root / "config" / "all_archs.yaml"
        if mapping_config is None:
            mapping_config = self.polaris_root / "config" / "wl2archmapping.yaml"
        if study_name is None:
            study_name = f"{workload_name}_study"
        if output_dir is None:
            output_dir = self.polaris_root / "output"
        
        # Build Polaris command
        cmd = [
            sys.executable, 
            str(self.polaris_root / "polaris.py"),
            "--archspec", str(arch_config),
            "--wlmapspec", str(mapping_config),
            "--study", study_name,
            "--odir", str(output_dir),
            "--filterwl", workload_name
        ]
        
        # Add workload instance filter if specified
        if workload_instance:
            cmd.extend(["--filterwli", workload_instance])
        
        # Add dry run flag if requested
        if dry_run:
            cmd.append("--dryrun")
        
        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)
        
        print(f"🚀 Running Polaris simulation...")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Workload: {workload_name}")
        if workload_instance:
            print(f"   Instance: {workload_instance}")
        print(f"   Study: {study_name}")
        print(f"   Output: {output_dir}")
        
        # Run the simulation
        try:
            result = subprocess.run(cmd, cwd=self.polaris_root, capture_output=False)
            return result.returncode
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
            return 1
    
    def analyze_results(self, study_name: str, output_dir: str = None) -> int:
        """Analyze simulation results using the built-in analyzer"""
        
        if output_dir is None:
            output_dir = self.polaris_root / "output"
        
        study_path = Path(output_dir) / study_name
        
        if not study_path.exists():
            print(f"❌ Study directory not found: {study_path}")
            return 1
        
        # Run the analyzer
        analyzer_script = self.polaris_root / "analyze_polaris_results.py"
        if not analyzer_script.exists():
            print(f"❌ Analyzer script not found: {analyzer_script}")
            return 1
        
        cmd = [sys.executable, str(analyzer_script), str(study_path)]
        
        print(f"📊 Analyzing results...")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.polaris_root)
            return result.returncode
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
            return 1
    
    def run_complete_workflow(self, generated_config_path: str,
                            workload_name: str, workload_instance: str = None,
                            dry_run_first: bool = True, analyze: bool = True) -> int:
        """Run complete workflow: setup, simulation, analysis"""
        
        print(f"🎯 Running complete Polaris workflow for {workload_name}")
        print(f"=" * 60)
        
        # Step 1: Create combined workload configuration
        print(f"📋 Step 1: Setting up workload configuration...")
        try:
            combined_config = self.create_combined_workload_config(generated_config_path)
            print(f"✅ Created combined config: {combined_config}")
        except Exception as e:
            print(f"❌ Failed to create combined config: {e}")
            return 1
        
        # Step 2: Run dry run if requested
        if dry_run_first:
            print(f"\n🧪 Step 2: Running dry run validation...")
            temp_wl_config = self.create_workload_config_with_combined(combined_config)
            result = self.run_simulation(
                workload_name=workload_name,
                workload_instance=workload_instance,
                study_name=f"{workload_name}_dryrun",
                dry_run=True,
                additional_args=["--wlspec", temp_wl_config]
            )
            
            if result != 0:
                print(f"❌ Dry run failed with exit code {result}")
                return result
            print(f"✅ Dry run completed successfully")
        
        # Step 3: Run actual simulation
        print(f"\n🚀 Step 3: Running actual simulation...")
        temp_wl_config = self.create_workload_config_with_combined(combined_config)
        study_name = f"{workload_name}_simulation"
        result = self.run_simulation(
            workload_name=workload_name,
            workload_instance=workload_instance,
            study_name=study_name,
            additional_args=["--wlspec", temp_wl_config, "--outputformat", "yaml", "--dump_stats_csv"]
        )
        
        if result != 0:
            print(f"❌ Simulation failed with exit code {result}")
            return result
        print(f"✅ Simulation completed successfully")
        
        # Step 4: Analyze results
        if analyze:
            print(f"\n📊 Step 4: Analyzing results...")
            result = self.analyze_results(study_name)
            
            if result != 0:
                print(f"⚠️  Analysis completed with warnings (exit code {result})")
            else:
                print(f"✅ Analysis completed successfully")
        
        print(f"\n🎉 Complete workflow finished!")
        print(f"   Study: {study_name}")
        print(f"   Results: {self.polaris_root}/output/{study_name}")
        print(f"   Analysis: {self.polaris_root}/analysis_output/")
        
        return 0
    
    def create_workload_config_with_combined(self, combined_config_path: str) -> str:
        """Create a temporary workload config file with the combined configuration"""
        # This is already handled by create_combined_workload_config
        return combined_config_path


def main():
    """Command line interface for Polaris simulation runner"""
    parser = argparse.ArgumentParser(
        description="Run Polaris simulations with generated workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow for a generated workload
  python run_polaris_simulation.py examples/configs/simple_cnn.yaml simplecnn

  # Run specific instance with dry run first
  python run_polaris_simulation.py examples/configs/simple_cnn.yaml simplecnn --instance simplecnn_batch4

  # Custom architecture and study name
  python run_polaris_simulation.py examples/configs/simple_cnn.yaml simplecnn \\
    --arch-config config/my_arch.yaml --study my_study

  # Just dry run for validation
  python run_polaris_simulation.py examples/configs/simple_cnn.yaml simplecnn --dry-run-only
        """
    )
    
    parser.add_argument('config', help='Generated workload configuration YAML file')
    parser.add_argument('workload', help='Workload name to simulate')
    parser.add_argument('--instance', '-i', help='Specific workload instance to run')
    parser.add_argument('--arch-config', '-a', help='Architecture configuration file')
    parser.add_argument('--mapping-config', '-m', help='Workload mapping configuration file')
    parser.add_argument('--study', '-s', help='Study name for the simulation')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--dry-run-only', action='store_true', help='Only run dry run validation')
    parser.add_argument('--no-dry-run', action='store_true', help='Skip dry run validation')
    parser.add_argument('--no-analysis', action='store_true', help='Skip result analysis')
    parser.add_argument('--polaris-root', default='.', help='Polaris root directory')
    parser.add_argument('--additional-args', nargs='*', help='Additional arguments to pass to Polaris')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    # Create runner
    runner = PolarisSimulationRunner(args.polaris_root)
    
    try:
        if args.dry_run_only:
            # Just run dry run
            combined_config = runner.create_combined_workload_config(args.config)
            temp_wl_config = runner.create_workload_config_with_combined(combined_config)
            result = runner.run_simulation(
                workload_name=args.workload,
                workload_instance=args.instance,
                arch_config=args.arch_config,
                mapping_config=args.mapping_config,
                study_name=args.study or f"{args.workload}_dryrun",
                output_dir=args.output_dir,
                dry_run=True,
                additional_args=(["--wlspec", temp_wl_config] + (args.additional_args or []))
            )
        else:
            # Run complete workflow
            result = runner.run_complete_workflow(
                generated_config_path=args.config,
                workload_name=args.workload,
                workload_instance=args.instance,
                dry_run_first=not args.no_dry_run,
                analyze=not args.no_analysis
            )
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Simulation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
