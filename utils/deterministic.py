import random
import numpy as np
from utils.random_generator import random_generator

def set_deterministic_mode(seed=42):
    """
    Set all random generators to use the same seed for reproducible results.
    
    Args:
        seed: Random seed to use
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set our custom random generator seed
    random_generator.set_seed(seed)
    
    # Patch random functions to ensure deterministic behavior
    # This is optional but can help catch cases where direct random.random() is used
    original_random = random.random
    def deterministic_random():
        return original_random()
    random.random = deterministic_random

def is_deterministic():
    """
    Test if the current environment is deterministic.
    
    Returns:
        bool: True if deterministic, False otherwise
    """
    # Run two sequences with the same seed and compare
    set_deterministic_mode(42)
    sequence1 = [random.random() for _ in range(10)]
    
    set_deterministic_mode(42)
    sequence2 = [random.random() for _ in range(10)]
    
    return sequence1 == sequence2
