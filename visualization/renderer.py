import pygame
import numpy as np
import math
import logging
from visualization.challenge_renderer import ChallengeRenderer

# Set up logging
logger = logging.getLogger(__name__)


class GridRenderer:
    """
    Handles rendering of the grid (zones, barriers, etc.) without creatures.
    This class is used by both the setup phase in main.py and the simulation phase.
    """
    def __init__(self, display_scale):
        """
        Initialize the grid renderer
        
        Args:
            display_scale: The scale factor for rendering grid cells
        """
        self.display_scale = display_scale
    
    def render_grid(self, screen, grid, params=None):
        """
        Render the grid with zones and barriers
        
        Args:
            screen: The pygame surface to render on
            grid: The grid object containing zone and barrier data
            params: Optional parameters for additional rendering options
        """
        scale = self.display_scale
        
        # Draw zones
        for x in range(grid.size[0]):
            for y in range(grid.size[1]):
                zone_type = grid.get_zone_type(x, y)
                if zone_type == 1:  # Safe zone
                    pygame.draw.rect(screen, (0, 180, 0, 64),
                                     (x * scale, y * scale, scale, scale))  # Green for safe
                elif zone_type == 2:  # Hazardous zone
                    pygame.draw.rect(screen, (180, 0, 0, 64),
                                     (x * scale, y * scale, scale, scale))  # Red for hazardous
        
        # Draw radiation (if enabled and params provided)
        if params and params.get('enable_radioactive_environment', False):
            for x in range(grid.size[0]):
                for y in range(grid.size[1]):
                    radiation = grid.data[x, y, 3]
                    if radiation > 0:
                        # Orange with transparency based on radiation level
                        intensity = int(radiation * 255)
                        pygame.draw.rect(screen, (intensity, intensity // 2, 0),
                                         (x * scale, y * scale, scale, scale))
        
        # Draw pheromones (blue) if enabled and params provided
        if params and params.get('show_pheromones', False):
            for x in range(grid.size[0]):
                for y in range(grid.size[1]):
                    pheromone = grid.data[x, y, 1]
                    if pheromone > 0:
                        # Draw as blue with transparency based on concentration
                        blue_val = int(pheromone * 255)
                        pygame.draw.rect(screen, (0, 0, blue_val),
                                         (x * scale, y * scale, scale, scale))
        
        # Draw barriers
        for bx, by in grid.barrier_locations:
            pygame.draw.rect(screen, (100, 100, 100),
                             (bx * scale, by * scale, scale, scale))


class CreatureRenderer:
    """
    Handles rendering of creatures on the grid.
    """
    def __init__(self, display_scale):
        """
        Initialize the creature renderer
        
        Args:
            display_scale: The scale factor for rendering creatures
        """
        self.display_scale = display_scale
    
    def render_creatures(self, screen, creatures, params):
        """
        Render creatures on the screen
        
        Args:
            screen: The pygame surface to render on
            creatures: List of creatures to render
            params: Parameters for creature rendering options
        """
        scale = self.display_scale
        
        # Calculate the maximum size a creature could be (for edge detection)
        max_possible_size = max(scale // 2, 2) + 2  # Base size + potential outline
        
        # Get screen dimensions
        screen_width, screen_height = screen.get_size()
        
        for creature in creatures:
            if not hasattr(creature, 'age') or creature.age <= params.get('max_age', float('inf')):
                # Set base size
                size = max(scale // 2, 2)
                
                # Calculate screen coordinates
                screen_x = int(creature.position[0] * scale)
                screen_y = int(creature.position[1] * scale)
                
                # Skip rendering if the creature would be partially off-screen
                if (screen_x - size < 0 or screen_x + size >= screen_width or
                    screen_y - size < 0 or screen_y + size >= screen_height):
                    continue
                
                # Get genome-based color for this creature
                creature_color = self.genome_to_color(creature.genome)
                
                # Creatures in safe zone get a yellow highlight
                if creature.in_safe_zone:
                    # Draw yellow outline
                    pygame.draw.circle(screen, (255, 255, 0),
                                       (screen_x, screen_y),
                                       size + 1)  # Outline
                    
                    # Safe zone creatures are slightly larger
                    size = min(scale - 1, int(scale * 0.75))
                
                # Draw the creature with its genome-based color
                pygame.draw.circle(screen, creature_color,
                                   (screen_x, screen_y),
                                   size)
                
                # Draw red outline around killers if that attribute exists
                if hasattr(creature, 'has_killed') and creature.has_killed:
                    pygame.draw.circle(screen, (255, 0, 0),
                                       (screen_x, screen_y),
                                       size + 2)
                
                # Draw direction line if enabled
                if params.get('show_direction_lines', True):
                    line_length = params.get('direction_line_length', 1)
                    if line_length > 0:
                        line_end = (int((creature.position[0] + creature.direction[0] * line_length) * scale),
                                    int((creature.position[1] + creature.direction[1] * line_length) * scale))
                        
                        # Only draw direction line if it's fully on screen
                        if (0 <= line_end[0] < screen_width and 0 <= line_end[1] < screen_height):
                            pygame.draw.line(screen, (255, 255, 255),
                                            (screen_x, screen_y),
                                            line_end, params.get('direction_line_thickness', 1))
    
    def genome_to_color(self, genome):
        """
        Convert a genome to an RGB color value
        
        Args:
            genome: The genome object to convert to a color
            
        Returns:
            Tuple of (r, g, b) values
        """
        # Extract a hash from the first 6 genes to determine the base color
        r_value = 0
        g_value = 0
        b_value = 0
        
        # Use the first 2 genes for red component
        for i in range(min(2, len(genome.genes))):
            gene_val = int(genome.genes[i].hex_value[:2], 16)
            r_value += gene_val
        
        # Use the next 2 genes for green component
        for i in range(2, min(4, len(genome.genes))):
            gene_val = int(genome.genes[i].hex_value[:2], 16)
            g_value += gene_val
        
        # Use the next 2 genes for blue component
        for i in range(4, min(6, len(genome.genes))):
            gene_val = int(genome.genes[i].hex_value[:2], 16)
            b_value += gene_val
        
        # Normalize values to 0-255 range and ensure they're not too dark
        r = min(255, max(40, r_value // 2))  # Ensure minimum brightness
        g = min(255, max(40, g_value // 2))
        b = min(255, max(40, b_value // 2))
        
        return (r, g, b)


class Renderer:
    """
    Main renderer class that handles the complete rendering process.
    Uses GridRenderer and CreatureRenderer for specific rendering tasks.
    """
    def __init__(self, params):
        """Initialize the renderer"""
        self.step = None
        self.generation = None
        self.params = params
        self.display_scale = params['display_scale']
        
        # Define border size
        self.border_size = 20  # Pixels for border around the grid
        
        # Calculate adjusted display size to include border
        grid_width = params['world_size'][0] * self.display_scale
        grid_height = params['world_size'][1] * self.display_scale
        display_width = grid_width + (self.border_size * 2)
        display_height = grid_height + (self.border_size * 2)
        self.display_size = (display_width, display_height)
        
        # Initialize component renderers
        self.grid_renderer = GridRenderer(self.display_scale)
        self.creature_renderer = CreatureRenderer(self.display_scale)
        self.challenge_renderer = ChallengeRenderer(self.display_scale)
        
        # Initialize Pygame
        pygame.init()
        self.world = pygame.display.set_mode(self.display_size)
        pygame.display.set_caption("DeepEvolution Simulator")
        self.clock = pygame.time.Clock()
        
        # Font for text
        self.font = pygame.font.SysFont(None, 24)
    
    def render_world(self, grid, creatures):
        """
        Render the world and all creatures
        
        Args:
            grid: The grid object containing world data
            creatures: List of creatures to render
        """
        # Clear screen
        self.world.fill((30, 30, 40))  # Dark blue-gray background
        
        # Create a surface for the grid with the exact grid dimensions
        grid_width = grid.size[0] * self.display_scale
        grid_height = grid.size[1] * self.display_scale
        grid_surface = pygame.Surface((grid_width, grid_height))
        grid_surface.fill((0, 0, 0))  # Black background for grid
        
        # Render challenge area if a challenge is selected and highlighting is enabled
        challenge_type = self.params.get('challenge', None)
        if challenge_type is not None and self.params.get('show_challenge_areas', False):
            self.challenge_renderer.render_challenge_area(grid_surface, grid, challenge_type, self.params)
        
        # Render grid elements (zones, barriers, etc.) on the grid surface
        self.grid_renderer.render_grid(grid_surface, grid, self.params)
        
        # Render creatures on the grid surface
        self.creature_renderer.render_creatures(grid_surface, creatures, self.params)
        
        # Draw a border around the grid
        border_rect = pygame.Rect(
            self.border_size - 2,  # Offset by 2 pixels to make border visible
            self.border_size - 2,
            grid_width + 4,  # Add 4 pixels to make border visible on all sides
            grid_height + 4
        )
        pygame.draw.rect(self.world, (100, 100, 100), border_rect, 2)  # Gray border, 2 pixels thick
        
        # Blit the grid surface onto the main surface with the border offset
        self.world.blit(grid_surface, (self.border_size, self.border_size))
        
        # Display generation and population info
        info_text = self.font.render(
            f"Gen: {self.generation} | Pop: {len(creatures)} | Step: {self.step}/{self.params['steps_per_generation']}",
            True, (255, 255, 255))
        self.world.blit(info_text, (10, 10))
        
        # Count creatures in safe zones
        safe_count = sum(1 for c in creatures if c.in_safe_zone)
        # Show safe zone status
        safe_percentage = safe_count / len(creatures) * 100 if creatures else 0
        safe_text = self.font.render(
            f"Safe Zones: {safe_count}/{len(creatures)} ({safe_percentage:.1f}%)",
            True, (0, 255, 0))
        self.world.blit(safe_text, (10, 35))
        
        # Display challenge name and highlighting status
        if challenge_type is not None:
            challenge_name = self.challenge_renderer.get_challenge_name(challenge_type)
            highlight_status = "ON" if self.params.get('show_challenge_areas', False) else "OFF"
            challenge_text = self.font.render(
                f"Challenge: {challenge_name} | Highlighting: {highlight_status}",
                True, (255, 255, 0))
            self.world.blit(challenge_text, (10, 60))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.params['fps'])
    
    def set_generation(self, generation):
        """Set the current generation number"""
        self.generation = generation
    
    def set_step(self, step):
        """Set the current step number"""
        self.step = step
    
    def handle_events(self):
        """
        Handle pygame events and return if simulation should continue
        
        Returns:
            Tuple of (running, paused) booleans
        """
        running = True
        paused = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        return running, paused
