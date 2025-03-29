import configparser
import json
import os

def load_cpp_config(ini_file):
    """
    Load C++ .ini configuration file and convert to Python parameters.
    
    Args:
        ini_file: Path to the C++ .ini file
        
    Returns:
        Dictionary with Python parameter equivalents
    """
    # Parameter mapping from C++ to Python
    parameter_mapping = {
        "population": "population_size",
        "stepsPerGeneration": "steps_per_generation",
        "maxGenerations": "max_generations",
        "maxNumberNeurons": "max_number_neurons",
        "genomeInitialLengthMin": "genome_initial_length_min",
        "genomeInitialLengthMax": "genome_initial_length_max",
        "genomeMaxLength": "genome_max_length",
        "pointMutationRate": "mutation_rate",
        "geneInsertionDeletionRate": "genes_insertion_deletion_rate",
        "deletionRatio": "deletion_ratio",
        "sexualReproduction": "sexual_reproduction",
        "chooseParentsByFitness": "choose_parents_by_fitness",
        "populationSensorRadius": "population_sensor_radius",
        "signalSensorRadius": "signal_sensor_radius",
        "responsivenessCurveKFactor": "responsiveness_curve_k_factor",
        "shortProbeBarrierDistance": "short_probe_barrier_distance",
        "barrierType": "barrier_type",
        "challenge": "challenge",
        "killEnable": "kill_enable",
        "sizeX": "world_size_x",
        "sizeY": "world_size_y",
        "signalLayers": "signal_layers",
        "displayScale": "display_scale",
        # Add more mappings as needed
    }
    
    # Type conversion functions
    type_converters = {
        "population_size": int,
        "steps_per_generation": int,
        "max_generations": int,
        "max_number_neurons": int,
        "genome_initial_length_min": int,
        "genome_initial_length_max": int,
        "genome_max_length": int,
        "mutation_rate": float,
        "genes_insertion_deletion_rate": float,
        "deletion_ratio": float,
        "sexual_reproduction": lambda x: x.lower() == "true",
        "choose_parents_by_fitness": lambda x: x.lower() == "true",
        "population_sensor_radius": float,
        "signal_sensor_radius": float,
        "responsiveness_curve_k_factor": int,
        "short_probe_barrier_distance": int,
        "barrier_type": int,
        "challenge": int,
        "kill_enable": lambda x: x.lower() == "true",
        "world_size_x": int,
        "world_size_y": int,
        "signal_layers": int,
        "display_scale": int,
        # Add more converters as needed
    }
    
    # Parse the INI file
    config = configparser.ConfigParser()
    config.read(ini_file)
    
    # Convert to Python parameters
    py_params = {}
    
    # Process the [params] section
    if "params" in config:
        for key, value in config["params"].items():
            # Create a case-insensitive mapping
            key_lower = key.lower()
            key_original = key
            
            # Find the matching parameter in the mapping (case-insensitive)
            matching_key = None
            for map_key in parameter_mapping:
                if map_key.lower() == key_lower:
                    matching_key = map_key
                    break
            
            if matching_key:
                py_key = parameter_mapping[matching_key]
                
                # Convert the value to the appropriate type
                if py_key in type_converters:
                    try:
                        py_value = type_converters[py_key](value)
                        py_params[py_key] = py_value
                    except Exception as e:
                        print(f"Error converting {key_original} to {py_key}: {e}")
                else:
                    py_params[py_key] = value
            else:
                # For parameters not in the mapping, use the original name
                print(f"Warning: Parameter '{key_original}' not in mapping, using original name")
                py_params[key_original] = value
    
    # Special handling for world_size
    if "world_size_x" in py_params and "world_size_y" in py_params:
        py_params["world_size"] = [py_params["world_size_x"], py_params["world_size_y"]]
        del py_params["world_size_x"]
        del py_params["world_size_y"]
    
    return py_params

def save_cpp_config_as_json(ini_file, json_file):
    """
    Convert a C++ .ini config file to a Python JSON config file.
    
    Args:
        ini_file: Path to the C++ .ini file
        json_file: Path to save the JSON file
    """
    py_params = load_cpp_config(ini_file)
    
    with open(json_file, "w") as f:
        json.dump(py_params, f, indent=4)

def create_default_config():
    """
    Create a default configuration file with parameters matching C++ defaults.
    
    Returns:
        Dictionary with default parameters
    """
    # Default parameters matching C++ biosim4.ini
    default_params = {
        "population_size": 300,
        "steps_per_generation": 1000,
        "max_generations": 100,
        "max_number_neurons": 20,
        "genome_initial_length_min": 24,
        "genome_initial_length_max": 48,
        "genome_max_length": 300,
        "mutation_rate": 0.001,
        "genes_insertion_deletion_rate": 0.0,
        "deletion_ratio": 0.5,
        "sexual_reproduction": True,
        "choose_parents_by_fitness": True,
        "population_sensor_radius": 2.5,
        "signal_sensor_radius": 2.5,
        "responsiveness_curve_k_factor": 2,
        "short_probe_barrier_distance": 4,
        "barrier_type": 0,
        "challenge": 0,
        "kill_enable": False,
        "world_size": [256, 256],
        "signal_layers": 1,
        "display_scale": 3,
        # Add more default parameters as needed
    }
    
    return default_params

def save_default_config(json_file="default_config.json"):
    """
    Save the default configuration to a JSON file.
    
    Args:
        json_file: Path to save the JSON file
    """
    default_params = create_default_config()
    
    with open(json_file, "w") as f:
        json.dump(default_params, f, indent=4)
