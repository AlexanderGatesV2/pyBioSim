import threading
import numpy as np
from core.signals import Signals

class ThreadSafeSignals(Signals):
    """
    Thread-safe wrapper for the Signals class.
    
    This class adds locks to ensure thread safety when multiple processes
    access the signals simultaneously.
    """
    def __init__(self, world_size, num_layers=1, params=None):
        """
        Initialize thread-safe pheromone signal layers.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
            num_layers: Number of independent signal layers to create
            params: Simulation parameters dictionary
        """
        super().__init__(world_size, num_layers, params)
        self.lock = threading.RLock()  # Reentrant lock for nested access
        self.layer_locks = [threading.Lock() for _ in range(num_layers)]
    
    def increment(self, layer, x, y):
        """Thread-safe version of increment"""
        with self.layer_locks[layer]:
            super().increment(layer, x, y)
    
    def fade(self, layer):
        """Thread-safe version of fade"""
        with self.layer_locks[layer]:
            super().fade(layer)
    
    def get_value(self, layer, x, y):
        """Thread-safe version of get_value"""
        with self.layer_locks[layer]:
            return super().get_value(layer, x, y)
    
    def get_gradient(self, layer, x, y, direction):
        """Thread-safe version of get_gradient"""
        with self.layer_locks[layer]:
            return super().get_gradient(layer, x, y, direction)
    
    @classmethod
    def from_signals(cls, signals):
        """
        Create a thread-safe signals object from an existing signals object.
        
        Args:
            signals: An existing Signals instance
            
        Returns:
            A new ThreadSafeSignals with copied data from the original signals
        """
        thread_safe_signals = cls(signals.size, signals.num_layers, signals.params)
        
        # Copy data from original signals
        with thread_safe_signals.lock:
            thread_safe_signals.data = np.copy(signals.data)
            thread_safe_signals.signal_increment_amount = signals.signal_increment_amount
            thread_safe_signals.signal_fade_amount = signals.signal_fade_amount
        
        return thread_safe_signals
