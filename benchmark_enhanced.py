import os
import time
import argparse
import logging
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from core.params import load_parameters
from core.grid import Grid
from core.signals import Signals
from core.simulator import Simulator
from core.parallel_simulator import ParallelSimulator
from core.enhanced_parallel_simulator import EnhancedParallelSimulator
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__, console_level=logging.INFO, file_level=logging.INFO)

def run_benchmark(params, simulator_type, num_steps=100):
    """
    Run a benchmark test with the given parameters.
    
    Args:
        params: Simulation parameters
        simulator_type: Type of simulator to use ('sequential', 'parallel', or 'enhanced')
        num_steps: Number of simulation steps to run
        
    Returns:
        dict: Benchmark results including average time per step and memory usage
    """
    # Initialize components
    grid = Grid(params['world_size'])
    signals = Signals(params['world_size'], params['signal_layers'], params=params)
    zone_manager = ZoneManager(grid, params)
    barrier_manager = BarrierManager(grid, params)
    
    # Create simulator
    if simulator_type == 'sequential':
        simulator = Simulator(params, grid, signals)
    elif simulator_type == 'parallel':
        simulator = ParallelSimulator(params, grid, signals)
    elif simulator_type == 'enhanced':
        simulator = EnhancedParallelSimulator(params, grid, signals)
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")
    
    simulator.zone_manager = zone_manager
    simulator.barrier_manager = barrier_manager
    grid.zone_manager = zone_manager
    
    # Initialize simulation
    simulator.initialize()
    
    # Run benchmark
    step_times = []
    
    for _ in range(num_steps):
        start_time = time.time()
        simulator.update()
        end_time = time.time()
        step_time = end_time - start_time
        step_times.append(step_time)
    
    # Calculate statistics
    avg_time = np.mean(step_times)
    min_time = np.min(step_times)
    max_time = np.max(step_times)
    std_time = np.std(step_times)
    
    # Clean up
    if hasattr(simulator, 'simulation_manager'):
        simulator.simulation_manager.close()
    elif hasattr(simulator, 'pool') and simulator.pool:
        simulator.pool.close()
        simulator.pool.join()
    
    return {
        'simulator_type': simulator_type,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'step_times': step_times
    }

def plot_results(results, output_file=None):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_file: Optional file path to save the plot
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot average time per step
    simulator_types = [r['simulator_type'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    
    # Bar chart for average time
    bars = ax1.bar(simulator_types, avg_times, yerr=std_times, capsize=10)
    ax1.set_title('Average Time per Step')
    ax1.set_xlabel('Simulator Type')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar, avg_time in zip(bars, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{avg_time:.6f}s',
                ha='center', va='bottom', rotation=0)
    
    # Plot step times
    for result in results:
        ax2.plot(result['step_times'], label=result['simulator_type'])
    
    ax2.set_title('Step Times')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.7)
    
    # Set x-axis to integers
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def calculate_speedup(results):
    """
    Calculate speedup and efficiency.
    
    Args:
        results: List of benchmark result dictionaries
        
    Returns:
        dict: Speedup and efficiency metrics
    """
    # Find sequential result
    sequential_time = None
    for result in results:
        if result['simulator_type'] == 'sequential':
            sequential_time = result['avg_time']
            break
    
    if sequential_time is None:
        return None
    
    # Calculate speedup and efficiency for each parallel result
    speedups = {}
    
    for result in results:
        if result['simulator_type'] != 'sequential':
            speedup = sequential_time / result['avg_time']
            efficiency = speedup / result['num_workers'] if 'num_workers' in result else None
            
            speedups[result['simulator_type']] = {
                'speedup': speedup,
                'efficiency': efficiency
            }
    
    return speedups

def main():
    """Run benchmarks comparing sequential, parallel, and enhanced parallel implementations."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark parallel vs sequential simulation')
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file path')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    parser.add_argument('--population', type=int, default=None, help='Override population size')
    parser.add_argument('--workers', type=int, default=None, help='Override number of workers')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--output', type=str, default=None, help='Output file for plot')
    parser.add_argument('--skip-sequential', action='store_true', help='Skip sequential benchmark')
    parser.add_argument('--skip-parallel', action='store_true', help='Skip original parallel benchmark')
    parser.add_argument('--skip-enhanced', action='store_true', help='Skip enhanced parallel benchmark')
    args = parser.parse_args()
    
    # Load parameters
    params = load_parameters(args.config)
    
    # Override parameters if specified
    if args.population:
        params['population_size'] = args.population
        
    if args.workers:
        params['num_workers'] = args.workers
    elif 'num_workers' not in params:
        params['num_workers'] = multiprocessing.cpu_count()
    
    if args.batch_size:
        params['batch_size'] = args.batch_size
    elif 'batch_size' not in params:
        params['batch_size'] = max(1, params['population_size'] // params['num_workers'])
    
    # Print benchmark configuration
    print(f"\n{'=' * 50}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'=' * 50}")
    print(f"Population size: {params['population_size']}")
    print(f"World size: {params['world_size']}")
    print(f"Number of steps: {args.steps}")
    print(f"Number of workers: {params['num_workers']}")
    print(f"Batch size: {params['batch_size']}")
    print(f"{'=' * 50}\n")
    
    # Run benchmarks
    results = []
    
    if not args.skip_sequential:
        print("Running sequential benchmark...")
        sequential_result = run_benchmark(params, 'sequential', num_steps=args.steps)
        sequential_result['num_workers'] = 1
        results.append(sequential_result)
        print(f"Sequential: {sequential_result['avg_time']:.6f} seconds per step")
    
    if not args.skip_parallel:
        print("\nRunning original parallel benchmark...")
        parallel_result = run_benchmark(params, 'parallel', num_steps=args.steps)
        parallel_result['num_workers'] = params['num_workers']
        results.append(parallel_result)
        print(f"Original Parallel: {parallel_result['avg_time']:.6f} seconds per step")
    
    if not args.skip_enhanced:
        print("\nRunning enhanced parallel benchmark...")
        enhanced_result = run_benchmark(params, 'enhanced', num_steps=args.steps)
        enhanced_result['num_workers'] = params['num_workers']
        results.append(enhanced_result)
        print(f"Enhanced Parallel: {enhanced_result['avg_time']:.6f} seconds per step")
    
    # Calculate speedup
    speedups = calculate_speedup(results)
    
    # Print results
    print(f"\n{'=' * 50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 50}")
    
    for result in results:
        print(f"{result['simulator_type'].capitalize()} time: {result['avg_time']:.6f} seconds per step (min: {result['min_time']:.6f}, max: {result['max_time']:.6f}, std: {result['std_time']:.6f})")
    
    if speedups:
        print("\nSpeedup:")
        for simulator_type, metrics in speedups.items():
            print(f"  {simulator_type.capitalize()}: {metrics['speedup']:.2f}x")
            if metrics['efficiency'] is not None:
                print(f"  {simulator_type.capitalize()} efficiency: {metrics['efficiency']:.2f}")
    
    print(f"{'=' * 50}")
    
    # Plot results
    if len(results) > 1:
        plot_results(results, args.output)

if __name__ == "__main__":
    main()
