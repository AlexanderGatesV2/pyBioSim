import numpy as np
import logging
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='serializable_creature_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SerializableCreature:
    """
    A serializable representation of a Creature that can be pickled for multiprocessing.
    
    This class contains only the essential data needed for creature updates in parallel processes.
    It does not contain complex objects like neural networks or genomes, which are replaced
    with serializable representations.
    """
    
    def __init__(self, creature=None):
        """
        Initialize a serializable creature, optionally from an existing Creature.
        
        Args:
            creature: Optional Creature object to convert from
        """
        # Basic properties
        self.id = 0
        self.position = (0, 0)
        self.direction = (0, 0)
        self.alive = True
        self.energy = 1000
        self.in_safe_zone = False
        self.has_killed = False
        
        # Enhanced state tracking
        self.birth_position = (0, 0)
        self.last_positions = []
        self.max_position_history = 10
        self.stuck_counter = 0
        
        # Metabolic and health factors
        self.health = 1.0
        self.metabolic_rate = 0.1
        self.energy_efficiency = 1.0
        
        # Behavioral traits
        self.aggression = 0.5
        self.curiosity = 0.5
        self.sociality = 0.5
        
        # Enhanced sensory capabilities
        self.longprobe_dist = 16
        self.recent_signals = []
        self.detected_creatures = []
        
        # Neural network characteristics
        self.responsiveness = 0.5
        self.oscPeriod = 34
        
        # Last movement
        self.last_move_offset = (0, 0)
        
        # Serializable neural network and genome
        self.serializable_brain = None
        self.serializable_genome = None
        
        # If a creature is provided, convert it
        if creature is not None:
            self.from_creature(creature)
    
    def from_creature(self, creature):
        """
        Convert a Creature object to a SerializableCreature.
        
        Args:
            creature: The Creature object to convert
            
        Returns:
            self: The updated SerializableCreature object
        """
        try:
            # Copy basic properties
            self.id = creature.id
            self.position = creature.position
            self.direction = creature.direction
            self.alive = creature.alive
            self.energy = creature.energy
            self.in_safe_zone = creature.in_safe_zone
            self.has_killed = creature.has_killed
            
            # Copy enhanced state tracking
            self.birth_position = creature.birth_position
            self.last_positions = creature.last_positions.copy() if hasattr(creature, 'last_positions') else []
            self.max_position_history = creature.max_position_history
            self.stuck_counter = creature.stuck_counter
            
            # Copy metabolic and health factors
            self.health = creature.health
            self.metabolic_rate = creature.metabolic_rate
            self.energy_efficiency = creature.energy_efficiency
            
            # Copy behavioral traits
            self.aggression = creature.aggression
            self.curiosity = creature.curiosity
            self.sociality = creature.sociality
            
            # Copy enhanced sensory capabilities
            self.longprobe_dist = creature.longprobe_dist
            self.recent_signals = creature.recent_signals.copy() if hasattr(creature, 'recent_signals') else []
            self.detected_creatures = creature.detected_creatures.copy() if hasattr(creature, 'detected_creatures') else []
            
            # Copy neural network characteristics
            self.responsiveness = creature.responsiveness
            self.oscPeriod = creature.oscPeriod
            
            # Copy last movement
            self.last_move_offset = creature.last_move_offset
            
            # Convert neural network and genome to serializable representations
            from agents.serializable_neural_network import SerializableNeuralNetwork
            from agents.serializable_genome import SerializableGenome
            
            self.serializable_brain = SerializableNeuralNetwork(creature.brain)
            self.serializable_genome = SerializableGenome(creature.genome)
            
            return self
        except Exception as e:
            logger.error(f"Error converting Creature to SerializableCreature: {e}", exc_info=True)
            raise
    
    def to_creature(self, params=None):
        """
        Convert a SerializableCreature back to a Creature.
        
        Args:
            params: Optional simulation parameters
            
        Returns:
            Creature: A new Creature object with the properties from this SerializableCreature
        """
        try:
            from agents.creature import Creature
            from agents.neural_network import NeuralNetwork
            from agents.genome import Genome
            
            # Create a new Creature with default values
            creature = Creature(params=params)
            
            # Update basic properties
            creature.id = self.id
            creature.position = self.position
            creature.direction = self.direction
            creature.alive = self.alive
            creature.energy = self.energy
            creature.in_safe_zone = self.in_safe_zone
            creature.has_killed = self.has_killed
            
            # Update enhanced state tracking
            creature.birth_position = self.birth_position
            creature.last_positions = self.last_positions.copy()
            creature.max_position_history = self.max_position_history
            creature.stuck_counter = self.stuck_counter
            
            # Update metabolic and health factors
            creature.health = self.health
            creature.metabolic_rate = self.metabolic_rate
            creature.energy_efficiency = self.energy_efficiency
            
            # Update behavioral traits
            creature.aggression = self.aggression
            creature.curiosity = self.curiosity
            creature.sociality = self.sociality
            
            # Update enhanced sensory capabilities
            creature.longprobe_dist = self.longprobe_dist
            creature.recent_signals = self.recent_signals.copy()
            creature.detected_creatures = self.detected_creatures.copy()
            
            # Update neural network characteristics
            creature.responsiveness = self.responsiveness
            creature.oscPeriod = self.oscPeriod
            
            # Update last movement
            creature.last_move_offset = self.last_move_offset
            
            # Convert serializable neural network and genome back to their original forms
            creature.genome = self.serializable_genome.to_genome(params)
            creature.brain = self.serializable_brain.to_neural_network(creature.genome, params)
            
            return creature
        except Exception as e:
            logger.error(f"Error converting SerializableCreature to Creature: {e}", exc_info=True)
            raise
    
    def update_from_creature(self, creature):
        """
        Update this SerializableCreature with the latest state from a Creature.
        
        Args:
            creature: The Creature object to update from
            
        Returns:
            self: The updated SerializableCreature object
        """
        try:
            # Update basic properties that might have changed
            self.position = creature.position
            self.direction = creature.direction
            self.alive = creature.alive
            self.energy = creature.energy
            self.in_safe_zone = creature.in_safe_zone
            self.has_killed = creature.has_killed
            
            # Update enhanced state tracking
            self.last_positions = creature.last_positions.copy() if hasattr(creature, 'last_positions') else []
            self.stuck_counter = creature.stuck_counter
            
            # Update health
            self.health = creature.health
            
            # Update neural network characteristics that might have changed
            self.responsiveness = creature.responsiveness
            self.oscPeriod = creature.oscPeriod
            self.longprobe_dist = creature.longprobe_dist
            
            # Update last movement
            self.last_move_offset = creature.last_move_offset
            
            # No need to update neural network or genome as they don't change during updates
            
            return self
        except Exception as e:
            logger.error(f"Error updating SerializableCreature from Creature: {e}", exc_info=True)
            raise
