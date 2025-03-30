import numpy as np


class Signals:
    # C++ uses uint8_t (0-255), Python uses float (0.0-1.0)
    CPP_SIGNAL_MAX = 255.0

    def __init__(self, world_size, num_layers=1, params=None):
        """
        Initialize the pheromone signal layers.
        
        Args:
            world_size: Tuple of (width, height) for the world grid
            num_layers: Number of independent signal layers to create
            params: Simulation parameters dictionary
        """
        self.size = world_size
        self.num_layers = num_layers
        self.data = np.zeros((num_layers, world_size[0], world_size[1]), dtype=float)
        self.params = params if params else {}
        
        # Get scaled parameters matching C++ defaults if available
        # C++ default signalAmount = 51
        cpp_signal_amount = self.params.get('signalAmount', 51)
        self.signal_increment_amount = float(cpp_signal_amount) / self.CPP_SIGNAL_MAX
        
        # C++ default fade is decrement by 1
        self.signal_fade_amount = 1.0 / self.CPP_SIGNAL_MAX

    def increment(self, layer, x, y):
        """
        Increment the signal level at the specified location and its neighbors,
        matching C++ logic (add half amount to neighbors).
        
        Args:
            layer: Signal layer index to modify
            x, y: Coordinates where the signal is emitted
        """
        center_x, center_y = int(x), int(y)
        amount = self.signal_increment_amount
        neighbor_amount = amount / 2.0

        if 0 <= center_x < self.size[0] and 0 <= center_y < self.size[1]:
            # Add to neighboring cells first (including center)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = center_x + dx, center_y + dy
                    if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                        current = self.data[layer, nx, ny]
                        self.data[layer, nx, ny] = min(1.0, current + neighbor_amount)
            
            # Add remaining half to center cell
            current = self.data[layer, center_x, center_y]
            self.data[layer, center_x, center_y] = min(1.0, current + neighbor_amount) # Add the other half

    def fade(self, layer):
        """
        Gradually reduce signal strength across the entire layer.
        
        Gradually reduce signal strength across the entire layer, matching C++ logic.
        
        Args:
            layer: Signal layer index to fade
        """
        amount = self.signal_fade_amount # Use scaled fade amount
        # Subtract from all cells with non-zero values
        mask = self.data[layer] > 0
        self.data[layer][mask] -= amount
        # Ensure no negative values
        self.data[layer] = np.maximum(0.0, self.data[layer])

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
