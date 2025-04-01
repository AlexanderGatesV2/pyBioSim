import numpy as np
import logging
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='serializable_neural_network_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SerializableNeuralNetwork:
    """
    A serializable representation of a NeuralNetwork that can be pickled for multiprocessing.
    
    This class contains only the essential data needed for neural network processing in parallel processes.
    It does not contain complex objects like functions or class methods, which are replaced
    with serializable representations.
    """
    
    def __init__(self, neural_network=None):
        """
        Initialize a serializable neural network, optionally from an existing NeuralNetwork.
        
        Args:
            neural_network: Optional NeuralNetwork object to convert from
        """
        # Constants
        self.SENSOR_MIN = 0.0
        self.SENSOR_MAX = 1.0
        self.NEURON_MIN = -1.0
        self.NEURON_MAX = 1.0
        self.ACTION_MIN = 0.0
        self.ACTION_MAX = 1.0
        
        # Neuron state
        self.neuron_outputs = []
        self.neuron_driven = []
        
        # Connection data
        self.connections = []
        
        # Network metrics
        self.active_internal_neurons = 0
        self.total_connections = 0
        
        # If a neural network is provided, convert it
        if neural_network is not None:
            self.from_neural_network(neural_network)
    
    def from_neural_network(self, neural_network):
        """
        Convert a NeuralNetwork object to a SerializableNeuralNetwork.
        
        Args:
            neural_network: The NeuralNetwork object to convert
            
        Returns:
            self: The updated SerializableNeuralNetwork object
        """
        try:
            # Copy constants
            self.SENSOR_MIN = neural_network.SENSOR_MIN
            self.SENSOR_MAX = neural_network.SENSOR_MAX
            self.NEURON_MIN = neural_network.NEURON_MIN
            self.NEURON_MAX = neural_network.NEURON_MAX
            self.ACTION_MIN = neural_network.ACTION_MIN
            self.ACTION_MAX = neural_network.ACTION_MAX
            
            # Copy neuron state
            self.neuron_outputs = [neuron['output'] for neuron in neural_network.neurons]
            self.neuron_driven = [neuron['driven'] for neuron in neural_network.neurons]
            
            # Copy connection data
            self.connections = []
            for conn in neural_network.connections:
                self.connections.append({
                    'source_type': conn['source_type'],
                    'source_num': conn['source_num'],
                    'sink_type': conn['sink_type'],
                    'sink_num': conn['sink_num'],
                    'weight': conn['weight']
                })
            
            # Copy network metrics
            self.active_internal_neurons = neural_network.active_internal_neurons
            self.total_connections = neural_network.total_connections
            
            return self
        except Exception as e:
            logger.error(f"Error converting NeuralNetwork to SerializableNeuralNetwork: {e}", exc_info=True)
            raise
    
    def to_neural_network(self, genome, params):
        """
        Convert a SerializableNeuralNetwork back to a NeuralNetwork.
        
        Args:
            genome: The genome for the neural network
            params: Simulation parameters
            
        Returns:
            NeuralNetwork: A new NeuralNetwork object with the properties from this SerializableNeuralNetwork
        """
        try:
            from agents.neural_network import NeuralNetwork
            
            # Create a new NeuralNetwork with the genome and params
            neural_network = NeuralNetwork(genome, params)
            
            # Replace the neurons with our serialized state
            neural_network.neurons = []
            for i in range(len(self.neuron_outputs)):
                neural_network.neurons.append({
                    'output': self.neuron_outputs[i],
                    'driven': self.neuron_driven[i]
                })
            
            # Replace the connections with our serialized connections
            neural_network.connections = []
            for conn in self.connections:
                neural_network.connections.append(conn.copy())
            
            # Update network metrics
            neural_network.active_internal_neurons = self.active_internal_neurons
            neural_network.total_connections = self.total_connections
            
            return neural_network
        except Exception as e:
            logger.error(f"Error converting SerializableNeuralNetwork to NeuralNetwork: {e}", exc_info=True)
            raise
    
    def update_from_neural_network(self, neural_network):
        """
        Update this SerializableNeuralNetwork with the latest state from a NeuralNetwork.
        
        Args:
            neural_network: The NeuralNetwork object to update from
            
        Returns:
            self: The updated SerializableNeuralNetwork object
        """
        try:
            # Update neuron state
            self.neuron_outputs = [neuron['output'] for neuron in neural_network.neurons]
            
            # No need to update connections as they don't change during updates
            
            return self
        except Exception as e:
            logger.error(f"Error updating SerializableNeuralNetwork from NeuralNetwork: {e}", exc_info=True)
            raise
