import unittest
import os
import json
import tempfile
from utils.config_compatibility import load_cpp_config, save_cpp_config_as_json, create_default_config
from core.params import load_parameters

class TestConfigCompatibility(unittest.TestCase):
    def setUp(self):
        # Create a temporary C++ config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cpp_config_path = os.path.join(self.temp_dir.name, "test_config.ini")
        
        with open(self.cpp_config_path, "w") as f:
            f.write("[params]\n")
            f.write("population = 500\n")
            f.write("stepsPerGeneration = 2000\n")
            f.write("maxGenerations = 50\n")
            f.write("sizeX = 300\n")
            f.write("sizeY = 200\n")
            f.write("sexualReproduction = true\n")
            f.write("challenge = 2\n")
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_cpp_config(self):
        """Test loading a C++ config file"""
        params = load_cpp_config(self.cpp_config_path)
        
        self.assertEqual(params["population_size"], 500)
        self.assertEqual(params["steps_per_generation"], 2000)
        self.assertEqual(params["max_generations"], 50)
        self.assertEqual(params["world_size"], [300, 200])
        self.assertTrue(params["sexual_reproduction"])
        self.assertEqual(params["challenge"], 2)
    
    def test_save_cpp_config_as_json(self):
        """Test converting a C++ config to JSON"""
        json_path = os.path.join(self.temp_dir.name, "test_config.json")
        save_cpp_config_as_json(self.cpp_config_path, json_path)
        
        # Check that the JSON file was created
        self.assertTrue(os.path.exists(json_path))
        
        # Load the JSON file and check its contents
        with open(json_path, "r") as f:
            params = json.load(f)
        
        self.assertEqual(params["population_size"], 500)
        self.assertEqual(params["steps_per_generation"], 2000)
        self.assertEqual(params["world_size"], [300, 200])
    
    def test_create_default_config(self):
        """Test creating a default configuration"""
        params = create_default_config()
        
        self.assertEqual(params["population_size"], 300)
        self.assertEqual(params["steps_per_generation"], 1000)
        self.assertEqual(params["world_size"], [256, 256])
        self.assertEqual(params["max_number_neurons"], 20)
        self.assertEqual(params["genome_initial_length_min"], 24)
        self.assertEqual(params["genome_initial_length_max"], 48)

if __name__ == "__main__":
    unittest.main()
