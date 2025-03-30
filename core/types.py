import math
import numpy as np
from enum import Enum


class Sensor(Enum):
    """
    Comprehensive sensor types available to creatures.
    Each sensor provides different environmental information.
    """
    LOC_X = 0                # Normalized X position in world (0-1)
    LOC_Y = 1                # Normalized Y position in world (0-1)
    BOUNDARY_DIST_X = 2      # Distance to nearest X boundary, normalized
    BOUNDARY_DIST_Y = 3      # Distance to nearest Y boundary, normalized
    BOUNDARY_DIST = 4        # Distance to nearest boundary in any direction, normalized
    GENETIC_SIM_FWD = 5      # Genetic similarity with creature in front (0-1)
    LAST_MOVE_DIR_X = 6      # X component of last movement direction (-1, 0, 1)
    LAST_MOVE_DIR_Y = 7      # Y component of last movement direction (-1, 0, 1)
    LONGPROBE_POP_FWD = 8    # Distance to nearest creature in forward direction
    LONGPROBE_BAR_FWD = 9    # Distance to nearest barrier in forward direction
    POPULATION = 10          # Local population density
    POPULATION_FWD = 11      # Population gradient in forward-backward axis
    POPULATION_LR = 12       # Population gradient in left-right axis
    OSC1 = 13                # Oscillator value (sine wave)
    AGE = 14                 # Normalized age (0-1 based on max age)
    BARRIER_FWD = 15         # Barrier detection in forward-reverse axis
    BARRIER_LR = 16          # Barrier detection in left-right axis
    RANDOM = 17              # Random value (0-1) each sensor read
    SIGNAL0 = 18             # Local pheromone concentration
    SIGNAL0_FWD = 19         # Pheromone gradient in forward-backward axis
    SIGNAL0_LR = 20          # Pheromone gradient in left-right axis
    # Removed Python-specific sensors: SAFE_ZONE, HAZARD_ZONE, RADIATION
    # Ensure this enum matches C++ sensors-actions.h up to NUM_SENSES
    # C++ NUM_SENSES = 21


class Action(Enum):
    """
    Comprehensive action types available to creatures.
    Each action allows creatures to interact with the environment in different ways.
    """
    MOVE_X = 0               # Move along X axis (value 0-1 maps to -1 to 1)
    MOVE_Y = 1               # Move along Y axis (value 0-1 maps to -1 to 1)
    MOVE_FORWARD = 2         # Move in current direction
    MOVE_RL = 3              # Rotate left/right (deprecated, use MOVE_LEFT/RIGHT)
    MOVE_RANDOM = 4          # Move in random direction
    SET_OSCILLATOR_PERIOD = 5 # Set period of internal oscillator
    SET_LONGPROBE_DIST = 6   # Set distance for long probe sensors
    SET_RESPONSIVENESS = 7   # Set overall responsiveness/activity level
    EMIT_SIGNAL0 = 8         # Emit pheromone signal
    MOVE_EAST = 9            # Move east (right)
    MOVE_WEST = 10           # Move west (left)
    MOVE_NORTH = 11          # Move north (up)
    MOVE_SOUTH = 12          # Move south (down)
    MOVE_LEFT = 13           # Move left relative to current direction
    MOVE_RIGHT = 14          # Move right relative to current direction
    MOVE_REVERSE = 15        # Move backward relative to current direction
    KILL_FORWARD = 16        # Attempt to kill creature in front


def normalize_vector(vector):
    """
    Normalize a vector to unit length.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        
    Returns:
        Tuple: Normalized vector with length 1, or (0, 0) if input has zero length
    """
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if length > 0:
        return (vector[0] / length, vector[1] / length)
    return (0, 0)


def vector_length(vector):
    """
    Calculate the length (magnitude) of a vector.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        
    Returns:
        Float: The length of the vector
    """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def calculate_genetic_similarity(genome1, genome2):
    """
    Calculate genetic similarity between two genomes.
    
    Similarity is measured as the proportion of genes that match exactly
    between the two genomes.
    
    Args:
        genome1: First genome to compare
        genome2: Second genome to compare
        
    Returns:
        Float: Similarity score between 0.0 (no similarity) and 1.0 (identical)
    """
    if len(genome1.genes) != len(genome2.genes):
        return 0.0

    matching_genes = 0
    for i in range(len(genome1.genes)):
        if genome1.genes[i].hex_value == genome2.genes[i].hex_value:
            matching_genes += 1

    return matching_genes / len(genome1.genes)
