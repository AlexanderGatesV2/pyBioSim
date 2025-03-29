import json
import os

DEFAULT_PARAMS = {
    'world_size': (128, 128), # Logical world size (width, height)
    'display_scale': 6, # Display scaling factor
    'genome_length': 32, # Number of genes (each gene is 8 hex digits)
    'mutation_rate': 0.01,
    'population_size': 1000,
    'steps_per_generation': 300, # Increased to give more time to find safe zones
    'fps': 300,
    'selection_method': 'zones', # Options: 'natural', 'random', 'zones'
    'enable_kill_neuron': False, # Whether the kill neuron is enabled
    'max_age': 300, # Maximum age of a creature
    'weight_divisor': 6000, # Divisor for neural connection weights
    'safe_zone_bonus': 10.0, # Energy bonus per step in safe zone
    'hazard_zone_penalty': 10.0, # Energy penalty per step in hazard zone
    'zone_size': 100, # Default size of zones when created
    'log_to_csv': True, # Whether to log generation statistics to CSV
    'log_folder': 'evolution_logs', # Folder to save CSV logs
    'enable_radioactive_environment': False, # Whether to enable radioactive environment
    'radiation_falloff_factor': 10.0, # Exponential falloff factor for radiation
    'radiation_switch_steps': 720, # Steps before radiation wall switches
    'show_direction_lines': True, # Toggle to show/hide direction lines
    'direction_line_length': 1, # Length of direction lines (set to 0 for very short lines)
    'direction_line_thickness': 1, # Thickness of direction lines
    'show_pheromones': True, # Toggle to show/hide pheromone trails
    'show_challenge_areas': False, # Toggle to show/hide challenge area highlighting
    'challenge_highlight_transparency': 40, # Transparency level for challenge area highlighting (0-255)
    'num_sensory_neurons': 18, # Position X/Y, genetic compatibility, borders, pheromones, zones, population gradients, etc.
    'num_internal_neurons': 18, # Internal neurons for processing
    'num_output_neurons': 6, # Movement X/Y, pheromone emission, responsiveness, oscillator control, kill
    "responsivenessCurveKFactor": 2,
    "longProbeDistance": 16,
    "shortProbeBarrierDistance": 4,
    "populationSensorRadius": 2.5,
    "barrierType": 0,
    "signal_layers": 1,
    'challenge': 6,  # Default challenge type
    'challenge_type': 6,  # Alternative key for consistency
}


def load_parameters(config_file=None):
    """
    Load simulation parameters from a config file or use defaults.
    
    This function loads parameters from a JSON config file if provided,
    otherwise it uses the default parameters. It also calculates any
    derived parameters based on the loaded values.
    
    Args:
        config_file: Path to a JSON configuration file (optional)
        
    Returns:
        Dictionary containing all simulation parameters
    """
    params = DEFAULT_PARAMS.copy()

    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_params = json.load(f)
                params.update(file_params)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default parameters")

    # Calculate derived parameters
    params['display_size'] = (params['world_size'][0] * params['display_scale'],
                              params['world_size'][1] * params['display_scale'])

    return params
