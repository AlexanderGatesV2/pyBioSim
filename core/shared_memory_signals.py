import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import logging
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='shared_memory_signals_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SharedMemorySignals:
    """
    A signals implementation that uses shared memory for parallel processing.
    
    This class provides a NumPy-like interface to a shared memory array,
    allowing multiple processes to access and modify the signals data efficiently.
    """
    
    def __init__(self, world_size, num_layers=1, params=None, existing_shm_name=None):
        """
        Initialize a shared memory signals object.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
            num_layers: Number of independent signal layers to create
            params: Simulation parameters dictionary
            existing_shm_name: Name of an existing shared memory block to use
        """
        self.size = tuple(world_size)
        self.num_layers = num_layers
        self.shape = (self.num_layers, self.size[0], self.size[1])
        self.dtype = np.float32
        self.params = params or {}
        
        # Calculate the size of the shared memory block
        self.nbytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize
        
        # Create or attach to shared memory
        if existing_shm_name is None:
            # Create new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
            self.owner = True
        else:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=existing_shm_name)
            self.owner = False
        
        # Create NumPy array backed by shared memory
        self.data = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        
        # Initialize data to zeros
        if self.owner:
            self.data.fill(0)
        
        # Initialize signal parameters
        self.signal_increment_amount = self.params.get('signal_increment_amount', 0.5)
        self.signal_fade_amount = self.params.get('signal_fade_amount', 0.01)
        
        # Create layer locks for thread safety
        self.layer_locks = [mp.Lock() for _ in range(self.num_layers)]
    
    def get_shm_name(self):
        """
        Get the name of the shared memory block.
        
        Returns:
            str: The name of the shared memory block
        """
        return self.shm.name
    
    def close(self):
        """
        Close the shared memory block.
        """
        self.shm.close()
        if self.owner:
            self.shm.unlink()
    
    def increment(self, layer, x, y):
        """
        Increment the signal value at a location.
        
        Args:
            layer: Signal layer index
            x: X coordinate
            y: Y coordinate
        """
        with self.layer_locks[layer]:
            # Ensure coordinates are within bounds
            if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                # Increment signal value, clamping to 1.0
                self.data[layer, x, y] = min(1.0, self.data[layer, x, y] + self.signal_increment_amount)
    
    def fade(self, layer):
        """
        Fade all signal values in a layer.
        
        Args:
            layer: Signal layer index
        """
        with self.layer_locks[layer]:
            # Subtract fade amount from all values in the layer
            self.data[layer] = np.maximum(0.0, self.data[layer] - self.signal_fade_amount)
    
    def get_value(self, layer, x, y):
        """
        Get the signal value at a location.
        
        Args:
            layer: Signal layer index
            x: X coordinate
            y: Y coordinate
            
        Returns:
            float: Signal value (0.0-1.0)
        """
        with self.layer_locks[layer]:
            # Ensure coordinates are within bounds
            if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                return float(self.data[layer, x, y])
            return 0.0
    
    def get_gradient(self, layer, x, y, direction):
        """
        Get the signal gradient at a location in a specific direction.
        
        Args:
            layer: Signal layer index
            x: X coordinate
            y: Y coordinate
            direction: Direction vector (dx, dy)
            
        Returns:
            float: Signal gradient (-1.0 to 1.0)
        """
        with self.layer_locks[layer]:
            # Ensure coordinates are within bounds
            if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                dx, dy = direction
                
                # Calculate neighboring coordinates
                x1 = max(0, min(self.size[0] - 1, int(x + dx)))
                y1 = max(0, min(self.size[1] - 1, int(y + dy)))
                
                # Calculate gradient
                return float(self.data[layer, x1, y1] - self.data[layer, x, y])
            return 0.0
    
    @classmethod
    def from_signals(cls, signals):
        """
        Create a shared memory signals object from an existing signals object.
        
        Args:
            signals: An existing Signals instance
            
        Returns:
            SharedMemorySignals: A new SharedMemorySignals with copied data from the original signals
        """
        # Create a new shared memory signals object
        shared_signals = cls(signals.size, num_layers=signals.num_layers, params=signals.params)
        
        # Copy data from original signals
        for layer in range(signals.num_layers):
            np.copyto(shared_signals.data[layer], signals.data[layer])
        
        # Copy signal parameters
        shared_signals.signal_increment_amount = signals.signal_increment_amount
        shared_signals.signal_fade_amount = signals.signal_fade_amount
        
        return shared_signals
