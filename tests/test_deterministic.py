import unittest
import random
import numpy as np
from utils.deterministic import set_deterministic_mode, is_deterministic
from utils.random_generator import random_generator

class TestDeterministicMode(unittest.TestCase):
    def test_deterministic_random(self):
        """Test that Python's random module is deterministic with our seed"""
        set_deterministic_mode(42)
        sequence1 = [random.random() for _ in range(10)]
        
        set_deterministic_mode(42)
        sequence2 = [random.random() for _ in range(10)]
        
        self.assertEqual(sequence1, sequence2)
    
    def test_deterministic_numpy(self):
        """Test that NumPy's random module is deterministic with our seed"""
        set_deterministic_mode(42)
        sequence1 = [np.random.random() for _ in range(10)]
        
        set_deterministic_mode(42)
        sequence2 = [np.random.random() for _ in range(10)]
        
        np.testing.assert_array_almost_equal(sequence1, sequence2)
    
    def test_deterministic_custom_generator(self):
        """Test that our custom random generator is deterministic"""
        set_deterministic_mode(42)
        sequence1 = [random_generator.random_float() for _ in range(10)]
        
        set_deterministic_mode(42)
        sequence2 = [random_generator.random_float() for _ in range(10)]
        
        self.assertEqual(sequence1, sequence2)
    
    def test_is_deterministic(self):
        """Test the is_deterministic function"""
        self.assertTrue(is_deterministic())

if __name__ == "__main__":
    unittest.main()
