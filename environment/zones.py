import numpy as np
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ZoneManager:
    """
    Centralized manager for all zone-related operations.
    
    This class handles creation and management of different zone types in the simulation grid,
    including safe zones (type 1) and hazard zones (type 2).
    """
    
    def __init__(self, grid, params):
        """
        Initialize the zone manager
        
        Args:
            grid: The simulation grid
            params: Simulation parameters
        """
        self.grid = grid
        self.params = params
        self.zone_size = params.get('zone_size', 50)
        
        # Track created zones for potential restoration
        self.zones = []

    def create_zone(self, pos, size=None, zone_type=1, track=True):
        """
        Create a circular zone of the specified type.
        
        Args:
            pos: (x, y) position for the center of the zone
            size: Diameter of the zone in grid cells (default: uses zone_size from params)
            zone_type: 1=safe, 2=hazard (default: 1)
            track: Whether to track this zone for potential restoration (default: True)
            
        Returns:
            Boolean: True if zone was created successfully
        """
        x, y = pos
        size = size if size is not None else self.zone_size
        radius = size // 2
        
        # Track this zone if requested
        if track:
            self.zones.append((x, y, size, zone_type))

        # Create the circular zone
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    world_x = min(self.grid.size[0] - 1, max(0, x + dx))
                    world_y = min(self.grid.size[1] - 1, max(0, y + dy))
                    self.grid.data[world_x, world_y, 2] = zone_type
                    
        logger.info(f"Created {'safe' if zone_type == 1 else 'hazard'} zone at ({x}, {y}) with radius {radius}")
        return True

    def create_directional_zone(self, direction, percentage=None, zone_type=1, track=True):
        """
        Create a zone covering a percentage of the map in a specific direction.
        
        Args:
            direction: 'left', 'right', 'top', 'bottom'
            percentage: 0-100 (percentage of map to cover) (default: 25%)
            zone_type: 1=safe, 2=hazard (default: 1)
            track: Whether to track this zone for potential restoration (default: True)
            
        Returns:
            Boolean: True if zone was created successfully
        """
        width, height = self.grid.size
        percentage = percentage if percentage is not None else 25
        
        # Track this zone if requested
        if track:
            self.zones.append((direction, percentage, zone_type, "directional"))

        # Create the directional zone based on direction
        if direction == 'left':
            boundary = int(width * percentage / 100)
            for x in range(boundary):
                for y in range(height):
                    self.grid.data[x, y, 2] = zone_type
            logger.info(f"Created {'safe' if zone_type == 1 else 'hazard'} zone covering left {percentage}% of map")

        elif direction == 'right':
            boundary = int(width * (1 - percentage / 100))
            for x in range(boundary, width):
                for y in range(height):
                    self.grid.data[x, y, 2] = zone_type
            logger.info(f"Created {'safe' if zone_type == 1 else 'hazard'} zone covering right {percentage}% of map")

        elif direction == 'top':
            boundary = int(height * percentage / 100)
            for y in range(boundary):
                for x in range(width):
                    self.grid.data[x, y, 2] = zone_type
            logger.info(f"Created {'safe' if zone_type == 1 else 'hazard'} zone covering top {percentage}% of map")

        elif direction == 'bottom':
            boundary = int(height * (1 - percentage / 100))
            for y in range(boundary, height):
                for x in range(width):
                    self.grid.data[x, y, 2] = zone_type
            logger.info(f"Created {'safe' if zone_type == 1 else 'hazard'} zone covering bottom {percentage}% of map")
        
        return True

    def place_random_zones(self, num_safe=3, num_hazard=2, min_size=None, max_size=None):
        """
        Place random safe and hazard zones in the world.
        
        Args:
            num_safe: Number of safe zones to place (default: 3)
            num_hazard: Number of hazard zones to place (default: 2)
            min_size: Minimum zone size (default: zone_size/2)
            max_size: Maximum zone size (default: zone_size*1.5)
            
        Returns:
            Boolean: True if zones were placed successfully
        """
        world_size = self.grid.size
        min_size = min_size if min_size is not None else self.zone_size // 2
        max_size = max_size if max_size is not None else int(self.zone_size * 1.5)
        
        # Place safe zones
        for _ in range(num_safe):
            x = random.randint(10, world_size[0] - 10)
            y = random.randint(10, world_size[1] - 10)
            size = random.randint(min_size, max_size)
            self.create_zone((x, y), size, 1)

        # Place hazard zones
        for _ in range(num_hazard):
            x = random.randint(10, world_size[0] - 10)
            y = random.randint(10, world_size[1] - 10)
            size = random.randint(min_size, max_size)
            self.create_zone((x, y), size, 2)
            
        logger.info(f"Placed {num_safe} safe zones and {num_hazard} hazard zones randomly")
        return True
        
    def restore_zones(self):
        """
        Restore all tracked zones after a grid reset.
        
        Returns:
            Boolean: True if zones were restored successfully
        """
        # Clear existing zone data
        self.grid.data[:, :, 2] = 0
        
        # Recreate all tracked zones
        for zone in self.zones:
            if len(zone) == 4 and zone[3] == "directional":
                # This is a directional zone
                direction, percentage, zone_type, _ = zone
                self.create_directional_zone(direction, percentage, zone_type, track=False)
            else:
                # This is a circular zone
                x, y, size, zone_type = zone
                self.create_zone((x, y), size, zone_type, track=False)
                
        logger.info(f"Restored {len(self.zones)} zones")
        return True
        
    def clear_zones(self):
        """
        Clear all zones from the grid and tracking.
        
        Returns:
            Boolean: True if zones were cleared successfully
        """
        self.grid.data[:, :, 2] = 0
        self.zones = []
        logger.info("Cleared all zones")
        return True
