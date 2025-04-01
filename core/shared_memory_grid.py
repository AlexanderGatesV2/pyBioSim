import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import logging
import math
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='shared_memory_grid_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SharedMemoryGrid:
    """
    A grid implementation that uses shared memory for parallel processing.
    
    This class provides a NumPy-like interface to a shared memory array,
    allowing multiple processes to access and modify the grid data efficiently.
    """
    
    def __init__(self, world_size, num_layers=4, existing_shm_name=None):
        """
        Initialize a shared memory grid.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
            num_layers: Number of data layers in the grid
            existing_shm_name: Name of an existing shared memory block to use
        """
        self.size = tuple(world_size)
        self.num_layers = num_layers
        self.shape = (self.size[0], self.size[1], self.num_layers)
        self.dtype = np.float32
        
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
        
        # Initialize barrier locations
        self.barrier_locations = []
        self.barrier_centers = []
        self.barrier_set = set()
        
        # Create a mask for non-barrier locations
        self.non_barrier_mask = np.ones((self.size[0], self.size[1]), dtype=bool)
        
        # Initialize death and move queues with locks
        self.death_queue_lock = mp.Lock()
        self.death_queue = []
        
        self.move_queue_lock = mp.Lock()
        self.move_queue = []
        
        # Zone manager reference (will be set by simulator)
        self.zone_manager = None
    
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
    
    def reset(self, keep_interactive_zones=False):
        """
        Reset the grid to its initial state.
        
        Args:
            keep_interactive_zones: Whether to keep interactive zones
        """
        with self.death_queue_lock:
            self.death_queue = []
        
        with self.move_queue_lock:
            self.move_queue = []
        
        # Reset grid data
        self.data.fill(0)
        
        # Reset barrier locations
        self.barrier_locations = []
        self.barrier_centers = []
        self.barrier_set = set()
        
        # Reset non-barrier mask
        self.non_barrier_mask = np.ones((self.size[0], self.size[1]), dtype=bool)
        
        # Reset zones if zone manager exists and not keeping interactive zones
        if self.zone_manager is not None and not keep_interactive_zones:
            self.zone_manager.reset()
    
    def is_empty_at(self, x, y):
        """
        Check if a location is empty (no creature or barrier).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if the location is empty
        """
        return (self.data[x, y, 0] == 0) and not self.is_barrier_at(x, y)
    
    def isOccupiedAt(self, x, y):
        """
        Check if a location is occupied by a creature.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if the location is occupied by a creature
        """
        return self.data[x, y, 0] > 0
    
    def is_barrier_at(self, x, y):
        """
        Check if a location is a barrier.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if the location is a barrier
        """
        return (x, y) in self.barrier_set
    
    def is_valid_move_target(self, x, y):
        """
        Check if a location is a valid move target (not a barrier and within bounds).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if the location is a valid move target
        """
        return (0 <= x < self.size[0] and 
                0 <= y < self.size[1] and 
                not self.is_barrier_at(x, y))
    
    def get_zone_type(self, x, y):
        """
        Get the zone type at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            int: Zone type (0=normal, 1=safe, 2=hazard)
        """
        return int(self.data[x, y, 2])
    
    def set_zone_type(self, x, y, zone_type):
        """
        Set the zone type at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            zone_type: Zone type (0=normal, 1=safe, 2=hazard)
        """
        self.data[x, y, 2] = zone_type
    
    def get_radiation_level(self, x, y):
        """
        Get the radiation level at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            float: Radiation level (0.0-1.0)
        """
        return float(self.data[x, y, 3])
    
    def set_radiation_level(self, x, y, level):
        """
        Set the radiation level at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            level: Radiation level (0.0-1.0)
        """
        self.data[x, y, 3] = level
    
    def get_pheromone_level(self, x, y):
        """
        Get the pheromone level at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            float: Pheromone level (0.0-1.0)
        """
        return float(self.data[x, y, 1])
    
    def set_pheromone_level(self, x, y, level):
        """
        Set the pheromone level at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            level: Pheromone level (0.0-1.0)
        """
        self.data[x, y, 1] = level
    
    def update_barrier_caches(self):
        """
        Update the barrier caches (barrier_set and non_barrier_mask).
        """
        # Update barrier set
        self.barrier_set = set(self.barrier_locations)
        
        # Update non-barrier mask
        self.non_barrier_mask = np.ones((self.size[0], self.size[1]), dtype=bool)
        for x, y in self.barrier_locations:
            self.non_barrier_mask[x, y] = False
    
    def find_empty_location(self):
        """
        Find a random empty location in the grid.
        
        Returns:
            tuple: (x, y) coordinates of an empty location
        """
        import random
        
        # Get indices of non-barrier locations
        non_barrier_indices = np.where(self.non_barrier_mask)
        non_barrier_coords = list(zip(non_barrier_indices[0], non_barrier_indices[1]))
        
        # Shuffle the coordinates
        random.shuffle(non_barrier_coords)
        
        # Find the first empty location
        for x, y in non_barrier_coords:
            if self.data[x, y, 0] == 0:
                return (x, y)
        
        # If no empty location is found, raise an exception
        raise Exception("No empty location found in grid")
    
    def queue_for_death(self, creature_id):
        """
        Queue a creature for death.
        
        Args:
            creature_id: ID of the creature to queue for death
        """
        with self.death_queue_lock:
            self.death_queue.append(creature_id)
    
    def queue_for_move(self, creature_id, new_pos):
        """
        Queue a creature for movement.
        
        Args:
            creature_id: ID of the creature to queue for movement
            new_pos: New position (x, y) for the creature
        """
        with self.move_queue_lock:
            self.move_queue.append((creature_id, new_pos))
    
    def process_death_queue(self, creatures_dict):
        """
        Process the death queue, removing creatures from the grid.
        
        Args:
            creatures_dict: Dictionary of creatures indexed by ID
        """
        with self.death_queue_lock:
            for creature_id in self.death_queue:
                if creature_id in creatures_dict:
                    creature = creatures_dict[creature_id]
                    x, y = int(creature.position[0]), int(creature.position[1])
                    
                    # Remove creature from grid
                    if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                        if self.data[x, y, 0] == creature_id:
                            self.data[x, y, 0] = 0
                    
                    # Mark creature as dead
                    creature.alive = False
            
            # Clear the death queue
            self.death_queue = []
    
    def process_move_queue(self, creatures_dict):
        """
        Process the move queue, moving creatures on the grid.
        
        Args:
            creatures_dict: Dictionary of creatures indexed by ID
        """
        with self.move_queue_lock:
            # Process each move in the queue
            for creature_id, new_pos in self.move_queue:
                if creature_id in creatures_dict:
                    creature = creatures_dict[creature_id]
                    old_x, old_y = int(creature.position[0]), int(creature.position[1])
                    new_x, new_y = int(new_pos[0]), int(new_pos[1])
                    
                    # Check if the new position is valid
                    if (0 <= new_x < self.size[0] and 
                        0 <= new_y < self.size[1] and 
                        not self.is_barrier_at(new_x, new_y)):
                        
                        # Check if the new position is empty
                        if self.data[new_x, new_y, 0] == 0:
                            # Remove creature from old position
                            if 0 <= old_x < self.size[0] and 0 <= old_y < self.size[1]:
                                if self.data[old_x, old_y, 0] == creature_id:
                                    self.data[old_x, old_y, 0] = 0
                            
                            # Add creature to new position
                            self.data[new_x, new_y, 0] = creature_id
                            
                            # Update creature position
                            creature.position = (new_x, new_y)
                            
                            # Calculate last move offset and direction change
                            dx = new_x - old_x
                            dy = new_y - old_y
                            creature.last_move_offset = (dx, dy)
                            
                            # Update direction based on movement
                            if dx != 0 or dy != 0:
                                magnitude = math.sqrt(dx * dx + dy * dy)
                                if magnitude > 0:
                                    creature.direction = (dx / magnitude, dy / magnitude)
                                
                                # Debug information for movement (commented out)
                                # print(f"Creature {creature_id} moved from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                                # print(f"  last_move_offset: {creature.last_move_offset}")
                                # print(f"  direction: {creature.direction}")
            
            # Clear the move queue
            self.move_queue = []
    
    def visit_neighborhood(self, loc, radius, callback_function):
        """
        Visit all locations in a neighborhood around a location.
        
        Args:
            loc: Center location (x, y)
            radius: Radius of the neighborhood
            callback_function: Function to call for each location in the neighborhood
        """
        x, y = loc
        
        # Calculate neighborhood bounds
        min_x = max(0, x - int(radius))
        max_x = min(self.size[0] - 1, x + int(radius))
        min_y = max(0, y - int(radius))
        max_y = min(self.size[1] - 1, y + int(radius))
        
        # Visit each location in the neighborhood
        for nx in range(min_x, max_x + 1):
            for ny in range(min_y, max_y + 1):
                # Calculate distance to center
                dx = nx - x
                dy = ny - y
                dist = (dx * dx + dy * dy) ** 0.5
                
                # Call callback function if within radius
                if dist <= radius:
                    callback_function((nx, ny))
    
    @classmethod
    def from_grid(cls, grid):
        """
        Create a shared memory grid from an existing grid.
        
        Args:
            grid: An existing Grid instance
            
        Returns:
            SharedMemoryGrid: A new SharedMemoryGrid with copied data from the original grid
        """
        # Create a new shared memory grid
        shared_grid = cls(grid.size, num_layers=grid.data.shape[2])
        
        # Copy data from original grid
        np.copyto(shared_grid.data, grid.data)
        
        # Copy barrier locations
        shared_grid.barrier_locations = grid.barrier_locations.copy()
        shared_grid.barrier_centers = grid.barrier_centers.copy()
        shared_grid.barrier_set = grid.barrier_set.copy()
        
        # Copy non-barrier mask
        np.copyto(shared_grid.non_barrier_mask, grid.non_barrier_mask)
        
        # Copy zone manager if it exists
        if hasattr(grid, 'zone_manager'):
            shared_grid.zone_manager = grid.zone_manager
        
        return shared_grid
