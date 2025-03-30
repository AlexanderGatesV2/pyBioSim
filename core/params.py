import json
import os
from utils.config_compatibility import load_cpp_config, create_default_config

# Default parameters that match C++ biosim4 defaults
DEFAULT_PARAMS = {
    'world_size': [256, 256],  # Logical world size [width, height]
    'display_scale': 3,  # Display scaling factor
    'genome_length': 32,  # Number of genes (each gene is 8 hex digits)
    'mutation_rate': 0.001,  # Match C++ pointMutationRate
    'population_size': 300,  # Match C++ population
    'steps_per_generation': 1000,  # Match C++ stepsPerGeneration
    'max_generations': 100,  # Match C++ maxGenerations
    'fps': 300,
    'selection_method': 'zones',  # Options: 'natural', 'random', 'zones'
    'enable_kill_neuron': False,  # Whether the kill neuron is enabled
    'max_age': 1000,  # Maximum age of a creature (match stepsPerGeneration)
    'weight_divisor': 8192.0,  # Divisor for neural connection weights (match C++)
    'safe_zone_bonus': 10.0,  # Energy bonus per step in safe zone
    'hazard_zone_penalty': 10.0,  # Energy penalty per step in hazard zone
    'zone_size': 100,  # Default size of zones when created
    'log_to_csv': True,  # Whether to log generation statistics to CSV
    'log_folder': 'evolution_logs',  # Folder to save CSV logs
    'enable_radioactive_environment': False,  # Whether to enable radioactive environment
    'radiation_falloff_factor': 10.0,  # Exponential falloff factor for radiation
    'radiation_switch_steps': 720,  # Steps before radiation wall switches
    'show_direction_lines': True,  # Toggle to show/hide direction lines
    'direction_line_length': 1,  # Length of direction lines (set to 0 for very short lines)
    'direction_line_thickness': 1,  # Thickness of direction lines
    'show_pheromones': True,  # Toggle to show/hide pheromone trails
    'show_challenge_areas': False,  # Toggle to show/hide challenge area highlighting
    'challenge_highlight_transparency': 40,  # Transparency level for challenge area highlighting (0-255)
    'num_sensory_neurons': 21,  # Match C++ NUM_SENSES (index 0-20)
    'num_internal_neurons': 20,  # Match C++ maxNumberNeurons
    'num_output_neurons': 17,  # Match C++ NUM_ACTIONS
    'responsiveness_curve_k_factor': 2,  # Match C++ responsivenessCurveKFactor
    'longprobe_dist': 16,  # Match C++ longProbeDistance
    'short_probe_barrier_distance': 4,  # Match C++ shortProbeBarrierDistance
    'population_sensor_radius': 2.5,  # Match C++ populationSensorRadius
    'signal_sensor_radius': 2.5,  # Match C++ signalSensorRadius
    'barrier_type': 0,  # Match C++ barrierType
    'signal_layers': 1,  # Match C++ signalLayers
    'challenge': 0,  # Match C++ challenge
    'genome_initial_length_min': 24,  # Match C++ genomeInitialLengthMin
    'genome_initial_length_max': 48,  # Match C++ genomeInitialLengthMax
    'genome_max_length': 300,  # Match C++ genomeMaxLength
    'genes_insertion_deletion_rate': 0.0,  # Match C++ geneInsertionDeletionRate
    'deletion_ratio': 0.5,  # Match C++ deletionRatio
    'sexual_reproduction': True,  # Match C++ sexualReproduction
    'choose_parents_by_fitness': True,  # Match C++ chooseParentsByFitness
    'kill_enable': False,  # Match C++ killEnable
}


def load_parameters(config_file=None):
    """
    Load simulation parameters from a config file or use defaults.
    
    This function loads parameters from a JSON or INI config file if provided,
    otherwise it uses the default parameters. It also calculates any
    derived parameters based on the loaded values.
    
    Args:
        config_file: Path to a configuration file (JSON or INI) (optional)
        
    Returns:
        Dictionary containing all simulation parameters
    """
    params = DEFAULT_PARAMS.copy()

    if config_file is None:
        return params

    try:
        # Check if the file exists
        if not os.path.exists(config_file):
            print(f"Warning: Config file '{config_file}' not found. Using defaults.")
            return params
        
        # Determine file type by extension
        _, ext = os.path.splitext(config_file)
        
        if ext.lower() == '.ini':
            # Load C++ INI file
            file_params = load_cpp_config(config_file)
        else:
            # Assume JSON file
            with open(config_file, 'r') as f:
                file_params = json.load(f)
        
        # Update parameters
        params.update(file_params)
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default parameters.")

    # Calculate derived parameters
    if isinstance(params['world_size'], list):
        params['display_size'] = (params['world_size'][0] * params['display_scale'],
                                params['world_size'][1] * params['display_scale'])
    else:
        params['display_size'] = (params['world_size'][0] * params['display_scale'],
                                params['world_size'][1] * params['display_scale'])

    return params
