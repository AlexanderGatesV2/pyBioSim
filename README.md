# pyBioSim

An evolutionary simulation of creatures with neural networks, compatible with biosim4.

## Description

pyBioSim is a simulation environment where virtual creatures with neural networks evolve over generations. The creatures navigate a 2D world, responding to environmental challenges through their neural networks, which are encoded by genomes that evolve through natural selection.

This project is a Python implementation of the [biosim4](https://github.com/davidrmiller/biosim4) C++ evolutionary simulation, with enhancements for visualization and interactivity.

## Features

- Neural network-controlled creatures
- Genetic algorithm with crossover and mutation
- Various environmental challenges
- Interactive visualization with pygame
- Customizable simulation parameters
- Data logging and analysis tools
- Compatibility with biosim4 C++ configuration files

## C++ Compatibility

This project is being updated to ensure compatibility with the original C++ biosim4 implementation. The goal is to faithfully recreate the C++ simulation behavior while maintaining the advantages of the Python architecture.

### Compatibility Status

- [ ] Gene and Genome Implementation
- [ ] Neural Network Implementation
- [ ] Movement and Sensor Implementation
- [ ] Survival and Reproduction
- [ ] Grid and Environment
- [ ] Integration and Verification
- [ ] Documentation and Finalization

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

During setup phase:
- **SPACE**: Start simulation
- **S**: Place safe zone
- **H**: Place hazard zone
- **C**: Cancel zone placement
- **Arrow Keys**: Create directional zones
- **1/2**: Decrease/Increase zone size
- **SHIFT+1/2**: Adjust directional zone percentage
- **0-3**: Select barrier type
- **F1**: Toggle help display
- **R**: Reset environment

During simulation:
- **SPACE**: Pause/Resume simulation
- **S**: Toggle challenge area highlighting
- **D**: Toggle direction lines
- **0-3**: Change barrier type
- **+/-**: Adjust simulation speed
- **G**: Force new generation (when paused)
- **R**: Reset simulation (when paused)
- **F1**: Toggle help display

## Configuration

The simulation can be customized through a JSON configuration file. See `config.json` for available parameters.

### C++ Configuration Files

pyBioSim can now read C++ biosim4 .ini configuration files directly:

```bash
# Run with a C++ biosim4 configuration file
python main.py --config biosim4.ini
```

This allows for direct comparison between the C++ and Python implementations using the same configuration.

## License

This project is open source and available under the MIT License.
