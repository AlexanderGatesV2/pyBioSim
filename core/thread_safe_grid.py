import threading
import numpy as np
from core.grid import Grid

class ThreadSafeGrid(Grid):
    """
    Thread-safe wrapper for the Grid class.
    
    This class adds locks to ensure thread safety when multiple processes
    access the grid simultaneously.
    """
    def __init__(self, world_size):
        """
        Initialize a thread-safe grid.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
        """
        super().__init__(world_size)
        self.lock = threading.RLock()  # Reentrant lock for nested access
        self.death_queue_lock = threading.Lock()
        self.move_queue_lock = threading.Lock()
        self.barrier_lock = threading.Lock()
    
    def reset(self, keep_interactive_zones=False):
        """Thread-safe version of reset"""
        with self.lock:
            super().reset(keep_interactive_zones)
    
    def is_empty_at(self, x, y):
        """Thread-safe version of is_empty_at"""
        with self.lock:
            return super().is_empty_at(x, y)
    
    def isOccupiedAt(self, x, y):
        """Thread-safe version of isOccupiedAt"""
        with self.lock:
            return super().isOccupiedAt(x, y)
    
    def is_barrier_at(self, x, y):
        """Thread-safe version of is_barrier_at"""
        with self.lock:
            return super().is_barrier_at(x, y)
    
    def is_valid_move_target(self, x, y):
        """Thread-safe version of is_valid_move_target"""
        with self.lock:
            return super().is_valid_move_target(x, y)
    
    def get_zone_type(self, x, y):
        """Thread-safe version of get_zone_type"""
        with self.lock:
            return super().get_zone_type(x, y)
    
    def set_zone_type(self, x, y, zone_type):
        """Thread-safe version of set_zone_type"""
        with self.lock:
            super().set_zone_type(x, y, zone_type)
    
    def get_radiation_level(self, x, y):
        """Thread-safe version of get_radiation_level"""
        with self.lock:
            return super().get_radiation_level(x, y)
    
    def set_radiation_level(self, x, y, level):
        """Thread-safe version of set_radiation_level"""
        with self.lock:
            super().set_radiation_level(x, y, level)
    
    def get_pheromone_level(self, x, y):
        """Thread-safe version of get_pheromone_level"""
        with self.lock:
            return super().get_pheromone_level(x, y)
    
    def set_pheromone_level(self, x, y, level):
        """Thread-safe version of set_pheromone_level"""
        with self.lock:
            super().set_pheromone_level(x, y, level)
    
    def update_barrier_caches(self):
        """Thread-safe version of update_barrier_caches"""
        with self.barrier_lock:
            super().update_barrier_caches()
    
    def find_empty_location(self):
        """Thread-safe version of find_empty_location"""
        with self.lock:
            return super().find_empty_location()
    
    def queue_for_death(self, creature_id):
        """Thread-safe version of queue_for_death"""
        with self.death_queue_lock:
            super().queue_for_death(creature_id)
    
    def queue_for_move(self, creature_id, new_pos):
        """Thread-safe version of queue_for_move"""
        with self.move_queue_lock:
            super().queue_for_move(creature_id, new_pos)
    
    def process_death_queue(self, creatures_dict):
        """Thread-safe version of process_death_queue"""
        with self.death_queue_lock:
            with self.lock:
                super().process_death_queue(creatures_dict)
    
    def process_move_queue(self, creatures_dict):
        """Thread-safe version of process_move_queue"""
        with self.move_queue_lock:
            with self.lock:
                super().process_move_queue(creatures_dict)
    
    def visit_neighborhood(self, loc, radius, callback_function):
        """Thread-safe version of visit_neighborhood"""
        with self.lock:
            super().visit_neighborhood(loc, radius, callback_function)

    @classmethod
    def from_grid(cls, grid):
        """
        Create a thread-safe grid from an existing grid.
        
        Args:
            grid: An existing Grid instance
            
        Returns:
            A new ThreadSafeGrid with copied data from the original grid
        """
        thread_safe_grid = cls(grid.size)
        
        # Copy data from original grid
        with thread_safe_grid.lock:
            thread_safe_grid.data = np.copy(grid.data)
            thread_safe_grid.barrier_locations = grid.barrier_locations.copy()
            thread_safe_grid.barrier_centers = grid.barrier_centers.copy()
            thread_safe_grid.barrier_set = grid.barrier_set.copy()
            thread_safe_grid.non_barrier_mask = np.copy(grid.non_barrier_mask)
            thread_safe_grid.interactive_zones = grid.interactive_zones.copy() if hasattr(grid, 'interactive_zones') else []
            
            # Copy zone_manager if it exists
            if hasattr(grid, 'zone_manager'):
                thread_safe_grid.zone_manager = grid.zone_manager
        
        return thread_safe_grid
