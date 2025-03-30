# Parallelization in pyBioSim

This document describes the parallelization implementation in pyBioSim, which aims to replicate the OpenMP parallelization from the original C++ biosim4 project.

## Overview

The parallelization is implemented using Python's `multiprocessing` module, which allows for true parallel execution across multiple CPU cores. The implementation focuses on parallelizing the creature update loop, which is the most computationally intensive part of the simulation.

## Components

The parallelization implementation consists of the following components:

1. **Thread-Safe Grid (`core/thread_safe_grid.py`)**: A thread-safe wrapper for the Grid class that adds locks to ensure thread safety when multiple processes access the grid simultaneously.

2. **Thread-Safe Signals (`core/thread_safe_signals.py`)**: A thread-safe wrapper for the Signals class that adds locks to ensure thread safety when multiple processes access the signals simultaneously.

3. **Parallel Simulator (`core/parallel_simulator.py`)**: A parallel implementation of the simulator that extends the base Simulator class to parallelize the creature update loop using Python's multiprocessing module.

4. **Configuration Options**: Added to `config.json` to control parallelization settings.

5. **Benchmark Script (`benchmark_parallel.py`)**: A script to benchmark the performance of the parallel implementation compared to the sequential implementation.

## Configuration Options

The following configuration options have been added to control parallelization:

-  `enable_parallelization` (boolean): Whether to use the parallel simulator. Default: `false`.
-  `num_workers` (integer): Number of worker processes to use. Default: number of CPU cores.
-  `batch_size` (integer): Number of creatures per batch. Default: `population_size / num_workers`.

Example configuration:

```json
{
	"enable_parallelization": true,
	"num_workers": 4,
	"batch_size": 50
}
```

## Usage

### Running with Parallelization

To run the simulation with parallelization, set `enable_parallelization` to `true` in your configuration file:

```json
{
	"enable_parallelization": true
}
```

Then run the simulation as usual:

```bash
python main.py
```

### Benchmarking

To benchmark the performance of the parallel implementation compared to the sequential implementation, use the `benchmark_parallel.py` script:

```bash
python benchmark_parallel.py
```

You can customize the benchmark with the following options:

-  `--config`: Path to the configuration file. Default: `config.json`.
-  `--steps`: Number of simulation steps to run. Default: `100`.
-  `--population`: Override the population size in the configuration.
-  `--workers`: Override the number of workers in the configuration.

Example:

```bash
python benchmark_parallel.py --steps 200 --population 1000 --workers 8
```

## Implementation Details

### Thread Safety

The thread-safe wrappers for Grid and Signals use Python's threading locks to ensure thread safety. The locks are used to protect critical sections of code that access shared data.

### Parallelization Strategy

The parallelization strategy is to divide the creatures into batches and process each batch in a separate process. The batch size is configurable and defaults to `population_size / num_workers`.

### Process Communication

The processes communicate by passing data between the main process and worker processes. Due to limitations in Python's multiprocessing module, we cannot directly share complex objects like Grid and Signals between processes. Instead, we pass the underlying NumPy arrays and recreate the objects in each worker process.

### Current Implementation Status

The current implementation includes:

1. Thread-safe wrappers for Grid and Signals
2. A parallel simulator class that extends the base simulator
3. Configuration options for controlling parallelization
4. A benchmark script for comparing sequential and parallel performance

However, there are some challenges that need to be addressed:

1. **Pickling Issues**: Python's multiprocessing module uses pickling to serialize objects for transfer between processes. Complex objects like thread locks and neural networks cannot be pickled.

2. **Creature Serialization**: The Creature class contains complex objects like neural networks that cannot be easily serialized. A more comprehensive solution would involve creating a serializable representation of creatures.

3. **State Synchronization**: When worker processes modify the grid and signals, these changes need to be synchronized back to the main process. The current implementation does not fully address this.

### Random Number Generation

Each process uses a process-specific random seed derived from the process ID and the simulation step to ensure deterministic behavior across processes.

## Performance Considerations

-  **Batch Size**: The batch size can significantly impact performance. A larger batch size reduces the overhead of process creation but may lead to load imbalance. A smaller batch size provides better load balancing but increases the overhead of process creation.

-  **Number of Workers**: The optimal number of workers depends on the number of CPU cores and the workload. In general, using a number of workers equal to the number of CPU cores provides the best performance.

-  **Memory Usage**: Each worker process requires memory for its own Python interpreter and data structures. This can lead to increased memory usage compared to the sequential implementation.

## Limitations

-  **Visualization**: The parallelization implementation does not parallelize the visualization code, which can be a bottleneck for large simulations.

-  **Process Creation Overhead**: There is an overhead associated with creating and managing multiple processes, which can reduce the performance gain for small simulations.

-  **Shared Memory**: The shared memory approach used by Python's `multiprocessing` module can be less efficient than the shared memory approach used by OpenMP in the C++ implementation.

-  **Pickling Limitations**: Python's multiprocessing module relies on pickling to serialize objects for transfer between processes. This creates challenges for complex objects like neural networks and thread locks.

-  **Partial Implementation**: Due to the challenges mentioned above, the current implementation falls back to sequential processing for the creature update loop. A more comprehensive solution would require significant changes to the codebase to make all objects serializable.

## Future Improvements

-  **Serializable Creatures**: Implement a serializable representation of creatures that can be passed between processes.

-  **Shared Memory Arrays**: Use shared memory arrays (e.g., `multiprocessing.Array`) for grid and signals data to avoid copying.

-  **Manager Process**: Use a manager process to coordinate access to shared data structures.

-  **Vectorization**: Use NumPy's vectorized operations to further optimize the code.

-  **Cython Integration**: Use Cython to compile performance-critical sections of the code to C.

-  **Dynamic Load Balancing**: Implement dynamic load balancing to better distribute the workload across processes.

-  **Hybrid Parallelization**: Combine multiprocessing with threading for a hybrid parallelization approach.
