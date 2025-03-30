#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import load_parameters

def test_load_ini_config():
    """Test loading a C++ INI config file"""
    params = load_parameters("sample_cpp_config.ini")
    
    print("Loaded parameters from INI file:")
    print(f"Population size: {params['population_size']}")
    print(f"Steps per generation: {params['steps_per_generation']}")
    print(f"World size: {params['world_size']}")
    print(f"Challenge: {params['challenge']}")
    print(f"Barrier type: {params['barrier_type']}")
    
    return params

def test_load_json_config():
    """Test loading a JSON config file"""
    params = load_parameters("sample_config.json")
    
    print("\nLoaded parameters from JSON file:")
    print(f"Population size: {params['population_size']}")
    print(f"Steps per generation: {params['steps_per_generation']}")
    print(f"World size: {params['world_size']}")
    print(f"Challenge: {params['challenge']}")
    print(f"Barrier type: {params['barrier_type']}")
    
    return params

def test_default_config():
    """Test loading default parameters"""
    params = load_parameters()
    
    print("\nLoaded default parameters:")
    print(f"Population size: {params['population_size']}")
    print(f"Steps per generation: {params['steps_per_generation']}")
    print(f"World size: {params['world_size']}")
    print(f"Challenge: {params['challenge']}")
    print(f"Barrier type: {params['barrier_type']}")
    
    return params

if __name__ == "__main__":
    print("Testing configuration loading...")
    
    ini_params = test_load_ini_config()
    json_params = test_load_json_config()
    default_params = test_default_config()
    
    # Verify that INI and JSON loading produce the same results
    assert ini_params['population_size'] == json_params['population_size']
    assert ini_params['steps_per_generation'] == json_params['steps_per_generation']
    assert ini_params['world_size'] == json_params['world_size']
    
    print("\nAll tests passed!")
