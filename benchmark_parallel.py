import os
import time
import argparse
import logging
import multiprocessing

from core.params import load_parameters
from core.grid import Grid
from core.signals import Signals
from core.simulator import Simulator
from core.parallel_simulator import ParallelSimulator
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__, console_level=logging.INFO, file_level=logging.INFO)

def run_benchmark(params, parallel=False, num_steps=100):
    """
    Run a benchmark test with the given parameters.
    
    Args:
        params: Simulation parameters
        parallel: Whether to use parallel simulator
        num_steps: Number of simulation steps to run
        
    Returns:
        float: Average time per step in seconds
    """
    # Initialize components
    grid = Grid(params['world_size'])
    signals = Signals(params['world_size'], params['signal_layers'], params=params)
    zone_manager = ZoneManager(grid, params)
    barrier_manager = BarrierManager(grid, params)
    
    # Create simulator
    if parallel:
        simulator = ParallelSimulator(params, grid, signals)
    else:
        simulator = Simulator(params, grid, signals)
    
    simulator.zone_manager = zone_manager
    simulator.barrier_manager = barrier_manager
    grid.zone_manager = zone_manager
    
    # Initialize simulation
    simulator.initialize()
    
    # Run benchmark
    start_time = time.time()
    
    for _ in range(num_steps):
        simulator.update()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_time_per_step = elapsed_time / num_steps
    
    # Clean up
    if parallel and simulator.pool:
        simulator.pool.close()
        simulator.pool.join()
    
    return avg_time_per_step

def main():
    """Run benchmarks comparing sequential and parallel implementations."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark parallel vs sequential simulation')
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file path')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    parser.add_argument('--population', type=int, default=None, help='Override population size')
    parser.add_argument('--workers', type=int, default=None, help='Override number of workers')
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
    
    # Print benchmark configuration
    print(f"\n{'=' * 50}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'=' * 50}")
    print(f"Population size: {params['population_size']}")
    print(f"World size: {params['world_size']}")
    print(f"Number of steps: {args.steps}")
    print(f"Number of workers: {params['num_workers']}")
    print(f"Batch size: {params.get('batch_size', params['population_size'] // params['num_workers'])}")
    print(f"{'=' * 50}\n")
    
    # Run sequential benchmark
    print("Running sequential benchmark...")
    seq_time = run_benchmark(params, parallel=False, num_steps=args.steps)
    print(f"Sequential: {seq_time:.6f} seconds per step")
    
    # Run parallel benchmark
    print("\nRunning parallel benchmark...")
    par_time = run_benchmark(params, parallel=True, num_steps=args.steps)
    print(f"Parallel: {par_time:.6f} seconds per step")
    
    # Calculate speedup
    speedup = seq_time / par_time
    efficiency = speedup / params['num_workers']
    
    print(f"\n{'=' * 50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 50}")
    print(f"Sequential time: {seq_time:.6f} seconds per step")
    print(f"Parallel time: {par_time:.6f} seconds per step")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.2f}")
    print(f"{'=' * 50}")
    
    print("\nNOTE: The parallel implementation is currently using sequential processing")
    print("for the creature update loop due to pickling issues with complex objects.")
    print("A full implementation would require serializable creature representations.")

if __name__ == "__main__":
    main()
