import numpy as np

class RandomGenerator:
    """
    A random number generator that matches the behavior of the C++ implementation.
    Each thread gets its own instance in the C++ version, but we'll use a single
    instance in Python since we're not multi-threading.
    """

    def __init__(self, seed=None, deterministic=False):
        """
        Initialize the random number generator.

        Args:
            seed: Optional seed for deterministic generation
            deterministic: Whether to use the seed for deterministic results
        """
        self.deterministic = deterministic

        if deterministic and seed is not None:
            # Use numpy's random for consistent behavior
            self.rng = np.random.RandomState(seed)
        else:
            # Use current time as seed for non-deterministic behavior
            import time
            current_time = int(time.time())
            self.rng = np.random.RandomState(current_time)

    def random_uint(self, min_val=0, max_val=None):
        """
        Generate a random unsigned integer.

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive), or full uint32 range if None

        Returns:
            Random integer in the specified range
        """
        if max_val is None:
            return self.rng.randint(0, 2 ** 32 - 1)

        return self.rng.randint(min_val, max_val + 1)

    def random_float(self):
        """
        Generate random float in range [0.0, 1.0]

        Returns:
            Random float between 0 and 1
        """
        return self.rng.random()

    def __call__(self, min_val=0, max_val=None):
        """
        Make the generator callable like its C++ counterpart.

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive), or full uint32 range if None

        Returns:
            Random integer in the specified range
        """
        return self.random_uint(min_val, max_val)


# Create a global instance for use throughout the simulation
random_generator = RandomGenerator()