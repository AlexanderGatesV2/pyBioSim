import time
import logging
import pygame
from agents.population import Population
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from environment.radiation import RadiationManager
from visualization.logger import Logger
from visualization.renderer import GridRenderer, CreatureRenderer
from visualization.challenge_renderer import ChallengeRenderer

# Set up logging
logger = logging.getLogger(__name__)


class CustomRenderer:
    """Simple renderer that handles its own pygame surface"""

    def __init__(self, params):
        self.params = params
        self.display_scale = params['display_scale']
        
        # Define border size
        self.border_size = 20  # Pixels for border around the grid
        
        # Create custom creature renderer that handles edge cases
        self.grid_renderer = GridRenderer(self.display_scale)
        
        # Use the updated CreatureRenderer from renderer.py
        # This version will skip rendering creatures that would be cut off at the edges
        self.creature_renderer = CreatureRenderer(self.display_scale)
        
        self.challenge_renderer = ChallengeRenderer(self.display_scale)

    def render_world(self, screen, grid, creatures):
        """Render the world with zones, barriers and creatures"""
        # Clear screen
        screen.fill((30, 30, 40))  # Dark blue-gray background
        
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
        # The updated CreatureRenderer will skip creatures that would be cut off at the edges
        self.creature_renderer.render_creatures(grid_surface, creatures, self.params)
        
        # Draw a border around the grid
        border_rect = pygame.Rect(
            self.border_size - 2,  # Offset by 2 pixels to make border visible
            self.border_size - 2,
            grid_width + 4,  # Add 4 pixels to make border visible on all sides
            grid_height + 4
        )
        pygame.draw.rect(screen, (100, 100, 100), border_rect, 2)  # Gray border, 2 pixels thick
        
        # Blit the grid surface onto the main surface with the border offset
        screen.blit(grid_surface, (self.border_size, self.border_size))

    # genome_to_color method removed - now handled by CreatureRenderer


class InteractiveSimulator:
    def __init__(self, params, grid, signals):
        """
        Initialize the interactive simulator with enhanced configuration tracking.

        Args:
            params (dict): Simulation configuration parameters
            grid (Grid): Simulation world grid
            signals (Signals): Pheromone/signal layer
        """
        self.params = params
        self.grid = grid
        self.signals = signals

        # Track first-time initialization
        self._first_run = True

        # Store initial configuration
        self._initial_config = {
            'population_size': params['population_size'],
            'initial_params': params.copy()
        }

        # Create simulation components
        self.population = Population(params['population_size'], params)
        self.zone_manager = ZoneManager(grid, params)
        self.barrier_manager = BarrierManager(grid, params)
        self.radiation_manager = RadiationManager(grid, params)
        self.logger = Logger(params)

        # Initialize other simulation state variables
        self.generation = 0
        self.step = 0
        self.running = True
        self.paused = False
        self.placing_zone = False
        self.zone_type = 1  # 1=safe, 2=hazard
        self.zone_size = params.get('zone_size', 50)
        self.directional_percentage = 10
        self.show_help = True

        # Pygame references will be initialized in run()
        self.display_scale = params['display_scale']
        
        # Define border size
        self.border_size = 20  # Pixels for border around the grid
        
        # Calculate adjusted display size to include border
        grid_width = params['world_size'][0] * self.display_scale
        grid_height = params['world_size'][1] * self.display_scale
        display_width = grid_width + (self.border_size * 2)
        display_height = grid_height + (self.border_size * 2)
        self.display_size = (display_width, display_height)
        
        self.screen = None
        self.clock = None
        self.font = None
        self.title_font = None
        self.instructions_window = None
        self.instructions_size = (350, 550)
        self.instructions_pos = None
        self.custom_renderer = None

    def initialize(self):
        """Initialize the simulation"""
        # Preserve initial configuration on first run
        if self._first_run:
            self._initial_config['initial_grid_data'] = self.grid.data.copy()
            self._initial_config['initial_signals_data'] = self.signals.data.copy()
            self._first_run = False

        # Reset grid
        self.grid.reset(keep_interactive_zones=True)

        # Recreate barriers based on initial configuration
        self.barrier_manager.create_barriers(
            self.params.get('barrierType', 0)
        )

        # Setup radiation environment if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.setup_radiation()

        # Reinitialize population with preserved size
        self.population = Population(
            size=self._initial_config['population_size'],
            params=self.params
        )
        self.population.initialize(self.grid)

        # Optional: Print initialization details for debugging
        print("Simulation reinitialized with preserved configuration")
        print(f"Population size: {len(self.population.creatures)}")
        print(f"Grid dimensions: {self.grid.size}")

        # Reset counters
        self.generation = 0
        self.step = 0

    def handle_events(self):
        """Handle pygame events and return control flags"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle pause
                    self.paused = not self.paused
                    logger.info(f"Simulation {'paused' if self.paused else 'resumed'}")

                # Challenge highlighting toggle
                elif event.key == pygame.K_s:
                    # Toggle challenge area highlighting
                    self.params['show_challenge_areas'] = not self.params.get('show_challenge_areas', False)
                    logger.info(f"Challenge area highlighting {'enabled' if self.params['show_challenge_areas'] else 'disabled'}")
                
                # Barrier type keys (0-3)
                elif event.key == pygame.K_0:
                    self.params['barrierType'] = 0
                    self.grid.reset()  # Clear existing barriers
                    self.barrier_manager.create_barriers(0)
                    logger.info("Barrier type set to 0: None")
                elif event.key == pygame.K_1:
                    self.params['barrierType'] = 1
                    self.grid.reset()  # Clear existing barriers
                    self.barrier_manager.create_barriers(1)
                    logger.info("Barrier type set to 1: Vertical bar in center")
                elif event.key == pygame.K_2:
                    self.params['barrierType'] = 2
                    self.grid.reset()  # Clear existing barriers
                    self.barrier_manager.create_barriers(2)
                    logger.info("Barrier type set to 2: Vertical bar in random location")
                elif event.key == pygame.K_3:
                    self.params['barrierType'] = 3
                    self.grid.reset()  # Clear existing barriers
                    self.barrier_manager.create_barriers(3)
                    logger.info("Barrier type set to 3: Five staggered blocks")

                # Simulation speed controls
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.params['fps'] = max(10, self.params['fps'] - 30)
                    logger.info(f"Simulation speed decreased to {self.params['fps']} fps")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    self.params['fps'] = min(1000, self.params['fps'] + 30)
                    logger.info(f"Simulation speed increased to {self.params['fps']} fps")

                # Feature toggles
                elif event.key == pygame.K_F1:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_d:
                    self.params['show_direction_lines'] = not self.params.get('show_direction_lines', True)
                    logger.info(f"Direction lines {'enabled' if self.params['show_direction_lines'] else 'disabled'}")

                # Force new generation
                elif event.key == pygame.K_g and self.paused:
                    self.step = self.params['steps_per_generation']  # This will trigger a new generation
                    logger.info("Forced new generation")

                # Reset key
                elif event.key == pygame.K_r and self.paused:
                    # Delete log files from previous run
                    self._delete_log_files()
                    # Re-initialize simulation
                    self.initialize()
                    logger.info("Simulation reset and log files deleted")

            # No more mouse events for placing zones

        return self.running, self.paused

    def create_zone(self, pos):
        """Create a zone at the specified position using ZoneManager"""
        # Use the ZoneManager to create the zone
        self.zone_manager.create_zone(pos, self.zone_size, self.zone_type)
        logger.info(f"Created {'safe' if self.zone_type == 1 else 'hazard'} zone at {pos} with size {self.zone_size}")

    def create_directional_zone(self, direction):
        """Create a directional zone using ZoneManager"""
        # Use the ZoneManager to create the directional zone
        self.zone_manager.create_directional_zone(direction, self.directional_percentage, self.zone_type)
        logger.info(f"Created {'safe' if self.zone_type == 1 else 'hazard'} directional zone ({direction}) with {self.directional_percentage}%")
        
    def _delete_log_files(self):
        """Delete log files from previous runs"""
        import os
        import glob
        
        # Get the log folder from params
        log_folder = self.params.get('log_folder', 'evolution_logs')
        
        # Delete CSV log files
        csv_files = glob.glob(os.path.join(log_folder, "*.csv"))
        for file in csv_files:
            try:
                os.remove(file)
                logger.info(f"Deleted log file: {file}")
            except Exception as e:
                logger.error(f"Failed to delete log file {file}: {e}")
        
        # Delete other log files
        log_files = glob.glob("*.log") + glob.glob("*/*.log")
        for file in log_files:
            try:
                os.remove(file)
                logger.info(f"Deleted log file: {file}")
            except Exception as e:
                logger.error(f"Failed to delete log file {file}: {e}")

    def render_help_window(self):
        """Render the help window with instructions"""
        # Fill window with semi-transparent background
        self.instructions_window.fill((0, 0, 50))  # Dark blue background

        y_offset = 10

        # Display zone placement info if active
        if self.placing_zone:
            zone_text = self.font.render(
                f"Placing {'Safe' if self.zone_type == 1 else 'Hazard'} Zone | Size: {self.zone_size}",
                True, (255, 255, 0))
            self.instructions_window.blit(zone_text, (10, y_offset))

            action_text = self.font.render("Click to place, C to cancel", True, (255, 255, 0))
            self.instructions_window.blit(action_text, (10, y_offset + 30))
            y_offset += 60
        else:
            # General instructions
            title = self.title_font.render("CONTROLS", True, (255, 255, 255))
            self.instructions_window.blit(title, (10, y_offset))
            y_offset += 30

            # Basic controls
            controls = [
                f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
                "SPACE: Pause/Resume simulation",
                "+/-: Adjust simulation speed",
                f"Speed: {self.params['fps']} fps",
                "",
                "Visual Controls:",
                "S: Toggle challenge highlighting",
                f"Challenge highlighting: {'ON' if self.params.get('show_challenge_areas', False) else 'OFF'}",
                "D: Toggle direction lines",
                f"Direction lines: {'ON' if self.params.get('show_direction_lines', True) else 'OFF'}",
                "",
                "Barrier Types:",
                "0: No barriers",
                "1: Vertical bar in center",
                "2: Vertical bar in random location",
                "3: Five staggered blocks",
                f"Current barrier type: {self.params.get('barrierType', 0)}",
                "",
                "Generation Control:",
                "G: Force new generation (when paused)",
                "R: Reset simulation and delete logs (when paused)",
                "",
                "F1: Toggle help display",
            ]

            for control in controls:
                if control == "":
                    y_offset += 5  # Less space for separation
                    continue

                control_text = self.font.render(control, True, (200, 200, 200))
                self.instructions_window.blit(control_text, (10, y_offset))
                y_offset += 22  # Reduced line spacing

        # Draw the instructions window
        self.screen.blit(self.instructions_window, self.instructions_pos)

    def update(self):
        """Update one simulation step"""
        if self.paused:
            return

        # Update all creatures
        self.population.update(self.grid, self.signals)

        # Update radiation if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.update_radiation()

        # Fade pheromones
        for layer in range(self.signals.num_layers):
            self.signals.fade(layer)

        # Increment step counter
        self.step += 1
        
        # Update params with current step for challenge visualization
        self.params['current_step'] = self.step

        # Check if generation is complete
        if self.step >= self.params['steps_per_generation']:
            self.end_generation()

    def end_generation(self):
        """Handle end of generation logic"""
        # Log generation stats
        self.logger.log_generation(self.generation, self.population.creatures)

        # Perform natural selection
        self.population.creatures = self.population.natural_selection_tournament(self.grid)

        # Increment generation counter and reset step counter
        self.generation += 1
        self.step = 0

        # Ultra-fast clearing using numpy operations and the non_barrier_mask
        # This is much faster than looping through each cell
        creature_layer = self.grid.data[:, :, 0]
        creature_layer[self.grid.non_barrier_mask] = 0
        
        # Place creatures in valid locations - use a batch approach for better performance
        empty_positions = []
        
        # Pre-generate a pool of empty positions
        for _ in range(min(len(self.population.creatures) * 2, 1000)):  # Generate more positions than needed
            try:
                pos = self.grid.find_empty_location()
                empty_positions.append(pos)
            except Exception:
                break  # Stop if we can't find any more empty positions
        
        # Place creatures using the pre-generated positions
        for i, creature in enumerate(self.population.creatures):
            if i < len(empty_positions):
                position = empty_positions[i]
            else:
                # Fallback if we run out of pre-generated positions
                position = self.grid.find_empty_location()
                
            # Set creature position and update grid
            creature.position = position
            self.grid.data[int(position[0]), int(position[1]), 0] = creature.id

    def render(self):
        """Render the current state of the simulation"""
        # Use our custom renderer that manages its own surface
        self.custom_renderer.render_world(self.screen, self.grid, self.population.creatures)

        # Display generation and step information
        info_text = self.font.render(
            f"Gen: {self.generation} | Pop: {len(self.population.creatures)} | " +
            f"Step: {self.step}/{self.params['steps_per_generation']} | " +
            f"Speed: {self.params['fps']} fps",
            True, (255, 255, 255))
        self.screen.blit(info_text, (10, 10))

        # Count creatures in safe zones
        safe_creatures = [c for c in self.population.creatures if c.in_safe_zone]
        safe_count = len(safe_creatures)
        safe_percentage = (safe_count / len(self.population.creatures) * 100
                           if self.population.creatures else 0)

        # Show safe zone status
        safe_text = self.font.render(
            f"Safe Zones: {safe_count}/{len(self.population.creatures)} ({safe_percentage:.1f}%)",
            True, (0, 255, 0))
        self.screen.blit(safe_text, (10, 35))
        
        # Display challenge name and highlighting status
        challenge_type = self.params.get('challenge', None)
        if challenge_type is not None:
            challenge_name = self.custom_renderer.challenge_renderer.get_challenge_name(challenge_type)
            highlight_status = "ON" if self.params.get('show_challenge_areas', False) else "OFF"
            challenge_text = self.font.render(
                f"Challenge: {challenge_name} | Highlighting: {highlight_status}",
                True, (255, 255, 0))
            self.screen.blit(challenge_text, (10, 60))

        # Show status if paused
        if self.paused:
            pause_text = self.font.render("PAUSED - Press SPACE to resume",
                                          True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(self.display_size[0] // 2, 20))
            self.screen.blit(pause_text, text_rect)

        # Show help window if enabled
        if self.show_help:
            self.render_help_window()

        # Update display
        pygame.display.flip()

    def run(self):
        """Run the simulation loop"""
        # Initialize pygame from scratch
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_size)
        pygame.display.set_caption("Evolution Simulator - Interactive Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 28)
        self.instructions_window = pygame.Surface(self.instructions_size)
        self.instructions_window.set_alpha(200)  # Semi-transparent
        self.instructions_pos = (10, self.display_size[1] - self.instructions_size[1] - 10)

        # Create our custom renderer that doesn't rely on existing pygame surfaces
        self.custom_renderer = CustomRenderer(self.params)

        # Initialize simulation
        self.initialize()

        try:
            # Main simulation loop
            while self.running:
                # Handle events
                run_flag, _ = self.handle_events()
                self.running = run_flag

                # Update simulation
                self.update()

                # Render current state
                self.render()

                # Control frame rate
                self.clock.tick(self.params['fps'])

        finally:
            # Clean up when simulation ends
            pygame.quit()
            self.logger.close()

__all__ = ['InteractiveSimulator']
