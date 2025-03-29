import numpy as np


class Signals:
    def __init__(self, world_size, num_layers=1):
        """
        Initialize the pheromone signal layers.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
            num_layers: Number of independent signal layers to create
        """
        self.size = world_size
        self.num_layers = num_layers
        self.data = np.zeros((num_layers, world_size[0], world_size[1]))

    def increment(self, layer, x, y, amount=0.2):
        """
        Increment the signal level at the specified location and its neighbors.
        
        Args:
            layer: Signal layer index to modify
            x, y: Coordinates where the signal is emitted
            amount: Amount to increase the signal (default: 0.2)
        """
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            # Add to center cell
            current = self.data[layer, int(x), int(y)]
            self.data[layer, int(x), int(y)] = min(1.0, current + amount)

            # Add to neighboring cells with smaller amount
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = int(x) + dx, int(y) + dy
                    if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                        current = self.data[layer, nx, ny]
                        self.data[layer, nx, ny] = min(1.0, current + (amount / 2))

    def fade(self, layer, amount=0.002):
        """
        Gradually reduce signal strength across the entire layer.
        
        Args:
            layer: Signal layer index to fade
            amount: Amount to decrease the signal (default: 0.002)
        """
        # Subtract from all cells with non-zero values
        mask = self.data[layer] > 0
        self.data[layer][mask] -= amount
        # Ensure no negative values
        self.data[layer] = np.maximum(0, self.data[layer])

    def get_value(self, layer, x, y):
        """
        Get the signal value at the specified location.
        
        Args:
            layer: Signal layer index to read from
            x, y: Coordinates to check
            
        Returns:
            Float value representing signal strength (0.0-1.0)
        """
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return self.data[layer, int(x), int(y)]
        return 0.0

    def get_gradient(self, layer, x, y, direction):
        """
        Get the signal gradient (difference) in the specified direction.
        
        Args:
            layer: Signal layer index to read from
            x, y: Coordinates to check
            direction: (dx, dy) tuple indicating direction to check
            
        Returns:
            Float value representing the gradient (-1.0 to 1.0)
        """
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return 0.0

        dx, dy = direction
        forward_x = min(self.size[0] - 1, max(0, int(x + dx)))
        forward_y = min(self.size[1] - 1, max(0, int(y + dy)))
        backward_x = min(self.size[0] - 1, max(0, int(x - dx)))
        backward_y = min(self.size[1] - 1, max(0, int(y - dy)))

        forward_value = self.data[layer, forward_x, forward_y]
        backward_value = self.data[layer, backward_x, backward_y]

        return forward_value - backward_value
