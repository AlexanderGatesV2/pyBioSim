import numpy as np
import random
import math

from utils.random_generator import random_generator


class Grid:
    def __init__(self, world_size):
        """
        Initialize the simulation grid.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
        """
        self.size = world_size
        # Grid data layers:
        # [0]=creature_id (positive int for creatures, -1 for barriers, 0 for empty)
        # [1]=pheromone_level (0.0-1.0)
        # [2]=zone_type (0=neutral, 1=safe, 2=hazardous)
        # [3]=radiation_level (0.0-1.0)
        self.data = np.zeros((world_size[0], world_size[1], 4))
        self.barrier_locations = []  # List of (x,y) barrier coordinates
        self.barrier_centers = []    # List of barrier center points for distance calculations
        self.interactive_zones = []  # Store interactively placed zones
        
        # Cache for faster barrier lookups
        self.barrier_set = set()
        
        # Create a mask for non-barrier cells (for fast clearing)
        self.non_barrier_mask = np.ones((world_size[0], world_size[1]), dtype=bool)

        # Queues for deferred operations to handle conflicts
        self.death_queue = []  # Creatures to remove at end of step
        self.move_queue = []   # Movement requests to process at end of step

    def reset(self, keep_interactive_zones=False):
        """
        Reset the grid to initial state.
        
        Args:
            keep_interactive_zones: If True, preserve user-placed zones
        """
        if keep_interactive_zones:
            # Save interactive zones before resetting
            saved_zones = self.interactive_zones.copy()
        
        # Clear all grid data
        self.data.fill(0)
        
        # Clear barrier tracking
        self.barrier_locations = []
        self.barrier_centers = []
        self.barrier_set = set()
        
        # Reset non-barrier mask
        self.non_barrier_mask.fill(True)

        # Restore zones if needed
        if keep_interactive_zones and hasattr(self, 'zone_manager'):
            # Use ZoneManager to restore zones if available
            self.zone_manager.restore_zones()
        elif keep_interactive_zones:
            # Legacy support - this will be removed once ZoneManager is fully integrated
            for zone in saved_zones:
                self.interactive_zones.append(zone)
                x, y, size, zone_type = zone
                # Set zone directly in grid data
                radius = size // 2
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            world_x = min(self.size[0] - 1, max(0, x + dx))
                            world_y = min(self.size[1] - 1, max(0, y + dy))
                            self.data[world_x, world_y, 2] = zone_type

    def is_empty_at(self, x, y):
        """
        Check if a cell is empty (no creature).
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Boolean: True if the cell is empty
        """
        return self.data[int(x), int(y), 0] == 0
        
    def isOccupiedAt(self, x, y):
        """
        Check if a cell is occupied by a creature.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Boolean: True if the cell is occupied by a creature
        """
        return self.data[int(x), int(y), 0] > 0

    def is_barrier_at(self, x, y):
        """
        Check if a cell contains a barrier.
        
        Barriers are marked with -1 in the creature layer.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Boolean: True if the cell contains a barrier
        """
        x_int, y_int = int(x), int(y)
        
        # Ensure coordinates are within grid bounds
        if not (0 <= x_int < self.size[0] and 0 <= y_int < self.size[1]):
            return False
        
        # Fast path: check grid data first (most common case)
        if self.data[x_int, y_int, 0] == -1:
            return True
            
        # Use cached barrier set for faster lookup
        return (x_int, y_int) in self.barrier_set
        
    def is_valid_move_target(self, x, y):
        """
        Check if a cell is a valid target for movement (empty and not a barrier).
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Boolean: True if the cell is a valid movement target
        """
        return self.is_empty_at(x, y) and not self.is_barrier_at(x, y)

    def get_zone_type(self, x, y):
        """
        Get the zone type at the specified location.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Integer: 0=neutral, 1=safe, 2=hazardous
        """
        return self.data[int(x), int(y), 2]

    def set_zone_type(self, x, y, zone_type):
        """
        Set the zone type at the specified location.
        
        Args:
            x, y: Coordinates to modify
            zone_type: 0=neutral, 1=safe, 2=hazardous
        """
        self.data[int(x), int(y), 2] = zone_type

    def get_radiation_level(self, x, y):
        """
        Get the radiation level at the specified location.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Float: Radiation level (0.0-1.0)
        """
        return self.data[int(x), int(y), 3]

    def set_radiation_level(self, x, y, level):
        """
        Set the radiation level at the specified location.
        
        Args:
            x, y: Coordinates to modify
            level: Radiation level (0.0-1.0)
        """
        self.data[int(x), int(y), 3] = level

    def get_pheromone_level(self, x, y):
        """
        Get the pheromone level at the specified location.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Float: Pheromone level (0.0-1.0)
        """
        return self.data[int(x), int(y), 1]

    def set_pheromone_level(self, x, y, level):
        """
        Set the pheromone level at the specified location.
        
        Args:
            x, y: Coordinates to modify
            level: Pheromone level (0.0-1.0)
        """
        self.data[int(x), int(y), 1] = level

    # create_zone method removed - now handled by ZoneManager
    # create_barriers method removed - now handled by BarrierManager

    def update_barrier_caches(self):
        """
        Update barrier caches for faster lookups.
        
        This method should be called after any changes to barrier locations.
        """
        # Update barrier set
        self.barrier_set = set(self.barrier_locations)
        
        # Update non-barrier mask
        self.non_barrier_mask.fill(True)
        for x, y in self.barrier_locations:
            self.non_barrier_mask[x, y] = False
    
    def find_empty_location(self):
        """
        Find a random empty location in the grid that is not a barrier.
        
        Returns:
            Tuple: (x, y) coordinates of an empty location
            
        Raises:
            Exception: If no empty locations are available
        """
        # Fast path: try random locations first
        for _ in range(50):  # Reduced number of attempts for better performance
            x = random_generator.random_uint(0, self.size[0] - 1)
            y = random_generator.random_uint(0, self.size[1] - 1)
            
            # Super fast check: if position is empty and not a barrier
            if self.data[x, y, 0] == 0 and (x, y) not in self.barrier_set:
                return (x, y)
        
        # Medium path: try with a grid-based approach
        spacing = max(1, min(self.size[0], self.size[1]) // 10)
        for x in range(spacing // 2, self.size[0], spacing):
            for y in range(spacing // 2, self.size[1], spacing):
                if self.data[x, y, 0] == 0 and (x, y) not in self.barrier_set:
                    return (x, y)
        
        # Slow path: scan systematically
        for x in range(0, self.size[0]):
            for y in range(0, self.size[1]):
                if self.data[x, y, 0] == 0 and (x, y) not in self.barrier_set:
                    return (x, y)

        # Last resort: just find any empty location
        for x in range(0, self.size[0]):
            for y in range(0, self.size[1]):
                if self.data[x, y, 0] == 0:
                    return (x, y)

        raise Exception("No empty locations available in grid")

    def queue_for_death(self, creature_id):
        """
        Queue a creature for death at end of step.
        
        Args:
            creature_id: ID of the creature to remove
        """
        if creature_id not in self.death_queue:
            self.death_queue.append(creature_id)

    def queue_for_move(self, creature_id, new_pos):
        """
        Queue a creature for movement at end of step.
        
        Args:
            creature_id: ID of the creature to move
            new_pos: (x, y) target position
        """
        self.move_queue.append((creature_id, new_pos))

    def process_death_queue(self, creatures_dict):
        """
        Process all queued deaths.
        
        Args:
            creatures_dict: Dictionary mapping creature IDs to creature objects
        """
        for creature_id in self.death_queue:
            if creature_id in creatures_dict:
                creature = creatures_dict[creature_id]
                x, y = int(creature.position[0]), int(creature.position[1])
                # Clear creature from grid
                if self.data[x, y, 0] == creature_id:
                    self.data[x, y, 0] = 0
                # Mark creature as dead
                creature.alive = False

        # Clear death queue
        self.death_queue.clear()

    def process_move_queue(self, creatures_dict):
        """
        Process all queued movements with conflict resolution.
        
        This method handles the case where multiple creatures try to move to the same position.
        It processes moves in random order to avoid bias, and only allows a move if the
        destination is still empty and not a barrier.
        
        Args:
            creatures_dict: Dictionary mapping creature IDs to creature objects
        """
        # Randomize the order of moves to avoid bias
        import random
        random.shuffle(self.move_queue)
        
        # Track which positions will be occupied after all moves
        reserved_positions = set()
        
        # First pass: identify valid moves (no conflicts)
        valid_moves = []
        for creature_id, new_pos in self.move_queue:
            # Skip if creature doesn't exist or is dead
            if creature_id not in creatures_dict or not creatures_dict[creature_id].alive:
                continue
                
            creature = creatures_dict[creature_id]
            new_x, new_y = int(new_pos[0]), int(new_pos[1])
            
            # Check if destination is valid (empty, not a barrier) and not reserved by another creature
            # Also ensure the destination is not already occupied by another creature
            if self.is_valid_move_target(new_x, new_y) and (new_x, new_y) not in reserved_positions and self.data[new_x, new_y, 0] == 0:
                # This is a valid move
                valid_moves.append((creature_id, new_pos))
                # Reserve this position
                reserved_positions.add((new_x, new_y))
        
        # Second pass: execute valid moves
        for creature_id, new_pos in valid_moves:
            creature = creatures_dict[creature_id]
            new_x, new_y = int(new_pos[0]), int(new_pos[1])
            
            # Clear old position
            old_x, old_y = int(creature.position[0]), int(creature.position[1])
            if self.data[old_x, old_y, 0] == creature_id:
                self.data[old_x, old_y, 0] = 0
            
            # Calculate discrete move offset
            dx = new_x - old_x
            dy = new_y - old_y
            
            # Update creature position and last move offset
            creature.position = new_pos
            creature.last_move_offset = (dx, dy) # Store the discrete offset
            
            # Update continuous direction if movement occurred
            if dx != 0 or dy != 0:
                magnitude = math.sqrt(dx * dx + dy * dy)
                if magnitude > 0:
                    # Keep internal direction normalized
                    creature.direction = (dx / magnitude, dy / magnitude)
                
                # Debug information for movement (commented out)
                # print(f"Creature {creature_id} moved from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                # print(f"  last_move_offset: {creature.last_move_offset}")
                # print(f"  direction: {creature.direction}")
            
            # Set new position in grid
            self.data[new_x, new_y, 0] = creature_id
            
            # Update creature's stuck counter if needed
            if hasattr(creature, 'stuck_counter'):
                creature.stuck_counter = 0
        
        # Clear move queue
        self.move_queue.clear()

    def visit_neighborhood(self, loc, radius, callback_function):
        """
        Visit all cells within a circular radius of the specified location.

        Args:
            loc: Tuple of (x, y) coordinates
            radius: The radius of the circular neighborhood
            callback_function: Function to call for each valid location within the radius
        """
        x, y = loc
        for dx in range(-min(int(radius), x), min(int(radius), self.size[0] - x)):
            extent_y = int(math.sqrt(radius * radius - dx * dx))
            for dy in range(-min(extent_y, y), min(extent_y, self.size[1] - y)):
                callback_function((x + dx, y + dy))
