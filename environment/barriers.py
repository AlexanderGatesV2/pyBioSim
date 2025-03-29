import random


class BarrierManager:
    def __init__(self, grid, params):
        """
        Initialize the barrier manager.
        
        Args:
            grid: The simulation grid
            params: Simulation parameters
        """
        self.grid = grid
        self.params = params

    def create_barriers(self, barrier_type):
        """
        Create barriers in the world based on the specified type.
        
        Args:
            barrier_type: Type of barrier pattern to create:
                0: No barriers
                1: Vertical bar in center
                2: Vertical bar in random location
                3: Five staggered blocks
        """
        # Clear existing barriers
        self.grid.barrier_locations = []
        self.grid.barrier_centers = []

        if barrier_type == 0:
            # Update barrier caches even if no barriers
            self.grid.update_barrier_caches()
            return

        # Helper function to draw a barrier directly to grid data
        def draw_barrier(min_x, min_y, max_x, max_y):
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if 0 <= x < self.grid.size[0] and 0 <= y < self.grid.size[1]:
                        self.grid.data[x, y, 0] = -1  # Mark as barrier
                        self.grid.barrier_locations.append((x, y))

        # Vertical bar in center
        if barrier_type == 1:
            min_x = self.grid.size[0] // 2
            max_x = min_x + 1
            min_y = self.grid.size[1] // 4
            max_y = min_y + self.grid.size[1] // 2
            draw_barrier(min_x, min_y, max_x, max_y)

        # Vertical bar in random location
        elif barrier_type == 2:
            min_x = random.randint(20, self.grid.size[0] - 20)
            max_x = min_x + 1
            min_y = random.randint(20, self.grid.size[1] // 2 - 20)
            max_y = min_y + self.grid.size[1] // 2
            draw_barrier(min_x, min_y, max_x, max_y)

        # Five blocks staggered
        elif barrier_type == 3:
            # Use original block sizes but with optimized rendering
            block_size_x = 2
            block_size_y = self.grid.size[0] // 3  # Original size

            # Helper function to draw a block directly to grid data
            def draw_box(min_x, min_y, max_x, max_y):
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if 0 <= x < self.grid.size[0] and 0 <= y < self.grid.size[1]:
                            self.grid.data[x, y, 0] = -1  # Mark as barrier
                            self.grid.barrier_locations.append((x, y))

            # Calculate positions for the five blocks
            x0 = self.grid.size[0] // 4 - block_size_x // 2
            y0 = self.grid.size[1] // 4 - block_size_y // 2
            x1 = x0 + block_size_x
            y1 = y0 + block_size_y

            # Draw the five blocks
            draw_box(x0, y0, x1, y1)  # Top-left
            
            x0 += self.grid.size[0] // 2
            x1 = x0 + block_size_x
            draw_box(x0, y0, x1, y1)  # Top-right
            
            y0 += self.grid.size[1] // 2
            y1 = y0 + block_size_y
            draw_box(x0, y0, x1, y1)  # Bottom-right
            
            x0 -= self.grid.size[0] // 2
            x1 = x0 + block_size_x
            draw_box(x0, y0, x1, y1)  # Bottom-left
            
            # Center block
            x0 = self.grid.size[0] // 2 - block_size_x // 2
            x1 = x0 + block_size_x
            y0 = self.grid.size[1] // 2 - block_size_y // 2
            y1 = y0 + block_size_y
            draw_box(x0, y0, x1, y1)  # Center

        # Update barrier caches for faster lookups
        self.grid.update_barrier_caches()
