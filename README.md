# pyBioSim

An evolutionary simulation of creatures with neural networks, compatible with biosim4.

## Description

pyBioSim is a simulation environment where virtual creatures with neural networks evolve over generations. The creatures navigate a 2D world, responding to environmental challenges through their neural networks, which are encoded by genomes that evolve through natural selection.

This project is a Python implementation of the [biosim4](https://github.com/davidrmiller/biosim4) C++ evolutionary simulation, with enhancements for visualization and interactivity.

## Features

-  Neural network-controlled creatures
-  Genetic algorithm with crossover and mutation
-  Various environmental challenges
-  Interactive visualization with pygame
-  Customizable simulation parameters
-  Data logging and analysis tools
-  Compatibility with biosim4 C++ configuration files

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
-  **K**: Toggle kill counter display
-  **0-3**: Change barrier type
-  **+/-**: Adjust simulation speed
-  **G**: Force new generation (when paused)
-  **R**: Reset simulation (when paused)
-  **F1**: Toggle help display

### Challenges

The simulation includes various environmental challenges that determine which creatures survive and reproduce:

-  **Circle Challenge (0)**: Creatures must stay inside a circular area in the top-left quadrant.
-  **Right Half (1)**: Creatures must stay in the right half of the arena.
-  **Right Quarter (2)**: Creatures must stay in the rightmost quarter of the arena.
-  **String (3)**: Creatures must have 2-3 neighbors within a radius of 1.5 cells.
-  **Center Weighted (4)**: Creatures must stay near the center, with higher scores for those closer to center.
-  **Center Unweighted (19)**: Creatures must stay near the center, with equal scores for all survivors.
-  **Corner (5)**: Creatures must stay near any corner of the arena.
-  **Corner Weighted (6)**: Creatures must stay near any corner, with higher scores for those closer to corners.
-  **Migrate Distance (7)**: Creatures are scored based on distance traveled from birth position.
-  **Center Sparse (8)**: Creatures must stay near center with a specific neighbor count.
-  **Left Eighth (9)**: Creatures must stay in the leftmost eighth of the arena.
-  **Radioactive Walls (10)**: Creatures must avoid walls that become radioactive.
-  **Against Any Wall (11)**: Creatures must touch any wall of the arena.
-  **Touch Any Wall (12)**: Creatures must have touched a wall during their lifetime.
-  **East West Eighths (13)**: Creatures must stay in the leftmost or rightmost eighth.
-  **Near Barrier (14)**: Creatures must stay near barriers.
-  **Pairs (15)**: Creatures must form exclusive pairs with specific neighbor configuration.
-  **Location Sequence (16)**: Creatures are scored based on number of locations visited.
-  **Altruism (17)**: Creatures in the northwest safe zone have higher scores.
-  **Altruism Sacrifice (18)**: Creatures in the northeast sacrifice zone are selected for kinship-based reproduction.

**Recent Fix**: The Center Unweighted challenge constant was changed from 4 to 19 to fix an issue where both Center Weighted and Center Unweighted challenges were using the same constant value (4). This ensures that the two different challenge types can be properly distinguished in the code.

### Kill Counter

The simulation includes a kill counter feature that tracks how many creatures are killed during each generation. This feature works when the kill neuron is enabled in your configuration.

-  Press **K** to toggle the kill counter display on/off
-  When enabled, the kill count appears in red text at the top-right of the screen
-  The kill count is also displayed in the help window (press F1)
-  The counter resets at the end of each generation
-  This feature helps track predatory behavior in the simulation

## Configuration

The simulation can be customized through a JSON configuration file. See `config.json` for available parameters.

### C++ Configuration Files

pyBioSim can read C++ biosim4 .ini configuration files directly:

```bash
# Run with a C++ biosim4 configuration file
python main.py --config biosim4.ini
```

This allows for direct comparison between the C++ and Python implementations using the same configuration.

## License

This project is open source and available under the MIT License.
