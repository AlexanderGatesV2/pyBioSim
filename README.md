# pyBioSim

An evolutionary simulation of creatures with neural networks.

## Description

pyBioSim is a simulation environment where virtual creatures with neural networks evolve over generations. The creatures navigate a 2D world, responding to environmental challenges through their neural networks, which are encoded by genomes that evolve through natural selection.

## Features

- Neural network-controlled creatures
- Genetic algorithm with crossover and mutation
- Various environmental challenges
- Interactive visualization with pygame
- Customizable simulation parameters
- Data logging and analysis tools

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

## License

This project is open source and available under the MIT License.
