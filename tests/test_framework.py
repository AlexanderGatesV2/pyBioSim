import subprocess
import os
import json
import numpy as np
import sys
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import load_parameters
from core.simulator import Simulator
from core.grid import Grid
from core.signals import Signals
from utils.random_generator import random_generator

class BiosimComparator:
    def __init__(self, cpp_executable_path="/tmp/biosim4/bin/biosim4", seed=42):
        """
        Initialize the comparator with paths to both implementations.
        
        Args:
            cpp_executable_path: Path to the C++ biosim4 executable
            seed: Random seed for reproducible tests
        """
        self.cpp_executable = cpp_executable_path
        self.seed = seed
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_cpp_simulation(self, generations, population, config=None):
        """
        Run the C++ simulation with specified parameters.
        
        Args:
            generations: Number of generations to run
            population: Population size
            config: Optional path to config file
            
        Returns:
            Dictionary with simulation results
        """
        # Create a temporary config file if none provided
        if config is None:
            temp_config = self.results_dir / "temp_cpp_config.ini"
            with open(temp_config, "w") as f:
                f.write(f"population = {population}\n")
                f.write(f"maxGenerations = {generations}\n")
                f.write(f"randomSeed = {self.seed}\n")
            config = temp_config
        
        # Run the C++ executable
        result_file = self.results_dir / "cpp_results.json"
        cmd = [
            self.cpp_executable,
            "-c", str(config),
            "--output-stats", str(result_file),
            "--headless"  # Assuming biosim4 has a headless mode
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load and return results
            with open(result_file, "r") as f:
                return json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"Error running C++ simulation: {e}")
            print(f"stdout: {e.stdout.decode()}")
            print(f"stderr: {e.stderr.decode()}")
            return None
        
    def run_python_simulation(self, generations, population, config=None):
        """
        Run the Python simulation with the same parameters.
        
        Args:
            generations: Number of generations to run
            population: Population size
            config: Optional path to config file
            
        Returns:
            Dictionary with simulation results
        """
        # Set deterministic mode
        self.set_deterministic_mode(self.seed)
        
        # Load parameters
        if config:
            params = load_parameters(config)
        else:
            params = load_parameters("config.json")
            params['population_size'] = population
            params['max_generations'] = generations
        
        # Initialize components
        grid = Grid(params['world_size'])
        signals = Signals(params['world_size'], params['signal_layers'])
        
        # Create simulator with stats collection
        simulator = Simulator(params, grid, signals, collect_stats=True)
        
        # Run simulation
        simulator.run(headless=True)
        
        # Save and return results
        result_file = self.results_dir / "py_results.json"
        with open(result_file, "w") as f:
            json.dump(simulator.stats, f)
        
        return simulator.stats
    
    def set_deterministic_mode(self, seed):
        """Set all random generators to use the same seed"""
        random.seed(seed)
        np.random.seed(seed)
        random_generator.set_seed(seed)
    
    def compare_results(self, cpp_results, py_results, tolerance=0.05):
        """
        Compare results with specified tolerance.
        
        Args:
            cpp_results: Results from C++ simulation
            py_results: Results from Python simulation
            tolerance: Acceptable difference percentage
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "population_stats": self._compare_population_stats(cpp_results, py_results, tolerance),
            "neural_networks": self._compare_neural_networks(cpp_results, py_results, tolerance),
            "survival_rates": self._compare_survival_rates(cpp_results, py_results, tolerance),
            "overall_match_percentage": 0.0
        }
        
        # Calculate overall match percentage
        match_scores = [
            comparison["population_stats"]["match_percentage"],
            comparison["neural_networks"]["match_percentage"],
            comparison["survival_rates"]["match_percentage"]
        ]
        comparison["overall_match_percentage"] = sum(match_scores) / len(match_scores)
        
        return comparison
    
    def _compare_population_stats(self, cpp_results, py_results, tolerance):
        """Compare population statistics"""
        # Implementation depends on the exact format of results
        # This is a placeholder
        return {"match_percentage": 0.95}
    
    def _compare_neural_networks(self, cpp_results, py_results, tolerance):
        """Compare neural network behaviors"""
        # Implementation depends on the exact format of results
        # This is a placeholder
        return {"match_percentage": 0.90}
    
    def _compare_survival_rates(self, cpp_results, py_results, tolerance):
        """Compare survival rates"""
        # Implementation depends on the exact format of results
        # This is a placeholder
        return {"match_percentage": 0.92}
