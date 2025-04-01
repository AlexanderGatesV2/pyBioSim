# pyBioSim

An evolutionary simulation of creatures with neural networks, compatible with biosim4.

## Description

pyBioSim is a simulation environment where virtual creatures with neural networks evolve over generations. The creatures navigate a 2D world, responding to environmental challenges through their neural networks, which are encoded by genomes that evolve through natural selection.

This project is a Python implementation of the [biosim4](https://github.com/davidrmiller/biosim4) C++ evolutionary simulation, with enhancements for visualization, interactivity, and parallelization.

## Features

-  Neural network-controlled creatures
-  Genetic algorithm with crossover and mutation
-  Various environmental challenges
-  Interactive visualization with pygame
-  Customizable simulation parameters
-  Data logging and analysis tools
-  Compatibility with biosim4 C++ configuration files
-  Parallelization for improved performance on multi-core systems

## C++ Compatibility

This project is being updated to ensure compatibility with the original C++ biosim4 implementation. The goal is to faithfully recreate the C++ simulation behavior while maintaining the advantages of the Python architecture.

### Compatibility Status

-  [x] Gene and Genome Implementation
-  [x] Neural Network Implementation
-  [x] Movement and Sensor Implementation
-  [x] Survival and Reproduction
-  [x] Grid and Environment
-  [x] Integration and Verification
-  [x] Documentation and Finalization

## Installation

### Quick Installation

#### On Linux/macOS:

```bash
# Make the script executable if needed
chmod +x install.sh

# Run the installation script
./install.sh
```

#### On Windows:

```
# Run the installation script
install.bat
```

The installation scripts will:

1. Optionally create a virtual environment
2. Install all required dependencies
3. Install the package in development mode

### Manual Installation

#### Using pip

```bash
# Install from the current directory
pip install .

# Or install in development mode
pip install -e .
```

#### Step by Step

1. Clone the repository
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate     # On Windows
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Running the Simulation

```bash
# Run with default parameters
python main.py

# Run with a custom configuration file
python main.py --config custom_config.json
```

### Controls

-  **SPACE**: Pause/Resume simulation
-  **S**: Toggle challenge area highlighting
-  **D**: Toggle direction lines
-  **0-3**: Change barrier type
-  **+/-**: Adjust simulation speed
-  **G**: Force new generation (when paused)
-  **R**: Reset simulation (when paused)
-  **F1**: Toggle help display

## Configuration

The simulation can be customized through a JSON configuration file. See `config.json` for available parameters.

### C++ Configuration Files

pyBioSim can read C++ biosim4 .ini configuration files directly:

```bash
# Run with a C++ biosim4 configuration file
python main.py --config biosim4.ini
```

This allows for direct comparison between the C++ and Python implementations using the same configuration.

## Parallelization

pyBioSim includes two parallelization implementations to improve performance on multi-core systems:

1. **Standard Parallelization**: A basic implementation using Python's multiprocessing module.
2. **Enhanced Parallelization**: An improved implementation with serializable creatures and shared memory.

### Configuration Options

The following configuration options control parallelization:

-  `enable_parallelization` (boolean): Whether to use parallelization. Default: `false`.
-  `use_enhanced_parallelization` (boolean): Whether to use the enhanced parallelization. Default: `false`.
-  `num_workers` (integer): Number of worker processes to use. Default: number of CPU cores.
-  `batch_size` (integer): Number of creatures per batch. Default: `population_size / num_workers`.

Example configuration:

```json
{
	"enable_parallelization": true,
	"use_enhanced_parallelization": true,
	"num_workers": 12,
	"batch_size": 250
}
```

### Running with Parallelization

To run the simulation with parallelization, set the appropriate options in your configuration file:

```bash
# Run with standard parallelization
python main.py --config parallel_config.json

# Run with enhanced parallelization
python main.py --config enhanced_parallel_config.json
```

### Benchmarking

To benchmark the performance of different implementations, use the benchmark scripts:

```bash
# Benchmark standard parallelization vs. sequential
python benchmark_parallel.py

# Benchmark enhanced parallelization vs. standard and sequential
python benchmark_enhanced.py
```

You can customize the benchmark with various options:

```bash
python benchmark_enhanced.py --steps 200 --population 1000 --workers 8 --output benchmark_results.png
```

### Implementation Details

#### Standard Parallelization

The standard parallelization implementation uses Python's multiprocessing module to parallelize the creature update loop. It includes:

-  Thread-safe wrappers for Grid and Signals
-  A parallel simulator class that extends the base simulator
-  Configuration options for controlling parallelization

#### Enhanced Parallelization

The enhanced parallelization implementation builds on the standard implementation with several improvements:

-  **Serializable Creatures**: Creatures can be serialized for transfer between processes.
-  **Shared Memory Arrays**: Grid and signals data are stored in shared memory.
-  **Simulation Manager**: A dedicated manager coordinates the parallel execution.
-  **Improved Performance**: Better performance and scalability, especially for large simulations.

#### Performance Considerations

-  **Batch Size**: A larger batch size reduces overhead but may lead to load imbalance.
-  **Number of Workers**: The optimal number of workers typically equals the number of CPU cores.
-  **Memory Usage**: Each worker process requires memory for its own Python interpreter and data structures.

## License

This project is open source and available under the MIT License.
