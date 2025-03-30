import os
import argparse
import pygame
import logging
import datetime
import csv

from utils.logging_config import configure_logging, get_logger, set_logging_level
from core.params import load_parameters
from core.grid import Grid
from core.signals import Signals
from visualization.interactive_simulator import InteractiveSimulator
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from visualization.renderer import GridRenderer

# Configure default logging
configure_logging()
logger = get_logger(__name__)
logging.getLogger('pygame').setLevel(logging.WARNING)  # Reduce pygame log verbosity


def main():
    """Main function to run the evolutionary simulation with interactive controls."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run evolutionary simulation with interactive controls')
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file path')
    args = parser.parse_args()

    # Load parameters
    params = load_parameters(args.config)
    log_folder = params['log_folder']

    # Create log directory if it doesn't exist
    if not os.path.exists(log_folder):
        try:
            os.makedirs(log_folder)
            logging.info(f"Log folder '{log_folder}' created.")
        except Exception as e:
            logging.error(f"Failed to create log folder '{log_folder}': {e}", exc_info=True)
            print(f"Failed to create log folder: {e}")  # Fallback console message

    # Configure logging
    log_file = os.path.join(log_folder, "simulation.log")
    configure_logging(log_file=log_file, console_level=logging.CRITICAL, file_level=logging.CRITICAL)

    # Initialize simulation components
    try:
        grid = Grid(params['world_size'])
        # Pass params to Signals constructor
        signals = Signals(params['world_size'], params['signal_layers'], params=params)
        zone_manager = ZoneManager(grid, params)
        barrier_manager = BarrierManager(grid, params)
        logging.info("Grid, Signals, ZoneManager, and BarrierManager initialized.")
    except Exception as e:
        logging.error("Failed to initialize Grid or Signals", exc_info=True)
        print(f"Failed to initialize Grid or Signals: {e}")
        return  # Exit if critical initialization fails

    # Initialize Pygame display and interface
    try:
        pygame.init()
        display_scale = params['display_scale']
        
        # Border size in pixels around the grid
        border_size = 20
        
        # Calculate display dimensions with border
        grid_width = params['world_size'][0] * display_scale
        grid_height = params['world_size'][1] * display_scale
        display_width = grid_width + (border_size * 2)
        display_height = grid_height + (border_size * 2)
        display_size = (display_width, display_height)
        
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption("Evolution Simulator - Setup Phase")
        clock = pygame.time.Clock()
        logging.info("Pygame initialized.")
    except Exception as e:
        logging.error("Failed to initialize Pygame", exc_info=True)
        print(f"Failed to initialize Pygame: {e}")
        return

    # Create simulator and connect components
    try:
        simulator = InteractiveSimulator(params, grid, signals)
        simulator.zone_manager = zone_manager
        simulator.barrier_manager = barrier_manager
        grid.zone_manager = zone_manager  # For reset operations
        logging.info("Simulator initialized with ZoneManager and BarrierManager.")
    except Exception as e:
        logging.error("Failed to initialize Simulator", exc_info=True)
        print(f"Failed to initialize Simulator: {e}")
        return

    # Initialize simulation state variables
    running = True
    placing_zone = False
    zone_type = 1  # 1=safe, 2=hazard
    zone_size = 50  # Default zone diameter
    directional_percentage = 25  # Default percentage for directional zones
    show_help = False  # Help display toggle
    simulation_started = False
    barrier_type = params.get('barrierType', 0)

    # Setup help/instructions window
    instructions_size = (350, 425)
    instructions_window = pygame.Surface(instructions_size)
    instructions_window.set_alpha(150)  # Semi-transparent background
    instructions_pos = (10, display_size[1] - instructions_size[1] - 10)

    # Setup fonts for UI text
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 28)

    # Initialize CSV logging for simulation data
    csv_info = setup_csv_logger(params)

    # Main setup loop - runs until simulation starts or user quits
    while running and not simulation_started:
        # Process user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logging.info("Received QUIT event. Exiting setup phase.")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Start the simulation
                    simulation_started = True
                    logging.info("SPACE key pressed. Starting simulation.")

                # Zone creation controls
                elif event.key == pygame.K_s:
                    # Switch to placing safe zones
                    placing_zone = True
                    zone_type = 1
                    logging.info("S key pressed. Now placing safe zones.")
                elif event.key == pygame.K_h:
                    # Switch to placing hazard zones
                    placing_zone = True
                    zone_type = 2
                    logging.info("H key pressed. Now placing hazard zones.")
                elif event.key == pygame.K_c:
                    # Cancel zone placement
                    placing_zone = False
                    logging.info("C key pressed. Cancelled zone placement.")

                # Directional zone creation controls
                elif event.key == pygame.K_LEFT:
                    zone_manager.create_directional_zone('left', directional_percentage, zone_type)
                    logging.info(f"LEFT key pressed. Created directional zone.")
                elif event.key == pygame.K_RIGHT:
                    zone_manager.create_directional_zone('right', directional_percentage, zone_type)
                    logging.info(f"RIGHT key pressed. Created directional zone.")
                elif event.key == pygame.K_UP:
                    zone_manager.create_directional_zone('top', directional_percentage, zone_type)
                    logging.info(f"UP key pressed. Created directional zone.")
                elif event.key == pygame.K_DOWN:
                    zone_manager.create_directional_zone('bottom', directional_percentage, zone_type)
                    logging.info(f"DOWN key pressed. Created directional zone.")

                # Zone size adjustment controls
                elif event.key == pygame.K_1:
                    # Decrease zone size or percentage
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        directional_percentage = max(10, directional_percentage - 10)
                        logging.info(f"1 key pressed with SHIFT. Directional zone percentage decreased to {directional_percentage}%")
                    else:
                        zone_size = max(20, zone_size - 10)
                        logging.info(f"1 key pressed. Zone size decreased to {zone_size}")
                elif event.key == pygame.K_2:
                    # Increase zone size or percentage
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        directional_percentage = min(90, directional_percentage + 10)
                        logging.info(f"2 key pressed with SHIFT. Directional zone percentage increased to {directional_percentage}%")
                    else:
                        zone_size = min(200, zone_size + 10)
                        logging.info(f"2 key pressed. Zone size increased to {zone_size}")

                # Barrier configuration controls
                elif event.key == pygame.K_0:
                    barrier_type = 0
                    grid.reset()  # Clear existing barriers
                    barrier_manager.create_barriers(barrier_type)
                    logging.info("0 key pressed. Barrier type: None")
                elif event.key == pygame.K_1:
                    barrier_type = 1
                    grid.reset()  # Clear existing barriers
                    barrier_manager.create_barriers(barrier_type)
                    logging.info("1 key pressed. Barrier type: Vertical bar in center")
                elif event.key == pygame.K_2:
                    barrier_type = 2
                    grid.reset()  # Clear existing barriers
                    barrier_manager.create_barriers(barrier_type)
                    logging.info("2 key pressed. Barrier type: Vertical bar in random location")
                elif event.key == pygame.K_3:
                    barrier_type = 3
                    grid.reset()  # Clear existing barriers
                    barrier_manager.create_barriers(barrier_type)
                    logging.info("3 key pressed. Barrier type: Five staggered blocks")

                # Help display toggle
                elif event.key == pygame.K_F1:
                    show_help = not show_help
                    logging.info(f"F1 key pressed. Help display {'shown' if show_help else 'hidden'}")

                # Environment reset control
                elif event.key == pygame.K_r:
                    # Reset simulation
                    grid.reset()
                    # Close existing CSV log file
                    close_csv_logger(csv_info)
                    # Create a new log file
                    csv_info = setup_csv_logger(params)
                    logging.info("R key pressed. Environment reset.")

            # Zone placement via mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN and placing_zone:
                try:
                    # Convert mouse position to world coordinates, accounting for border
                    mouse_pos = pygame.mouse.get_pos()
                    adjusted_x = mouse_pos[0] - border_size
                    adjusted_y = mouse_pos[1] - border_size
                    
                    # Only create zones for clicks within the grid area
                    if 0 <= adjusted_x < grid_width and 0 <= adjusted_y < grid_height:
                        world_x = int(adjusted_x / display_scale)
                        world_y = int(adjusted_y / display_scale)
                        zone_manager.create_zone((world_x, world_y), zone_size, zone_type)
                        logging.info(f"Created zone at ({world_x}, {world_y})")
                except Exception as e:
                    logging.error(f"Error creating zone: {e}", exc_info=True)

        # Render the setup screen
        screen.fill((30, 30, 40))  # Dark blue-gray background
        
        # Create grid surface with exact dimensions
        grid_width = grid.size[0] * display_scale
        grid_height = grid.size[1] * display_scale
        grid_surface = pygame.Surface((grid_width, grid_height))
        grid_surface.fill((0, 0, 0))  # Black background for grid
        
        # Render the grid content
        grid_renderer = GridRenderer(display_scale)
        grid_renderer.render_grid(grid_surface, grid)
        
        # Draw border around the grid area
        border_rect = pygame.Rect(
            border_size - 2,  # Offset by 2 pixels to make border visible
            border_size - 2,
            grid_width + 4,  # Add 4 pixels to make border visible on all sides
            grid_height + 4
        )
        pygame.draw.rect(screen, (100, 100, 100), border_rect, 2)  # Gray border, 2 pixels thick
        
        # Place the grid on the main screen with border offset
        screen.blit(grid_surface, (border_size, border_size))

        # Display instruction text at the top
        info_text = font.render("Press SPACE to start simulation | Press F1 for Help", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))

        # Show zone placement information when active
        if placing_zone:
            zone_text = font.render(
                f"Placing {'Safe' if zone_type == 1 else 'Hazard'} Zone | Size: {zone_size}",
                True, (255, 255, 0))
            screen.blit(zone_text, (10, 40))

            action_text = font.render("Click to place, C to cancel", True, (255, 255, 0))
            screen.blit(action_text, (10, 70))

        # Display help window when toggled on
        if show_help:
            render_help_window(screen, instructions_window, instructions_pos,
                               font, title_font, placing_zone, zone_type,
                               zone_size, directional_percentage, barrier_type)

        # Update the screen
        pygame.display.flip()
        clock.tick(30)  # Cap at 30 fps for setup phase

    # Either start the simulation or exit
    if simulation_started:
        # Save user-configured parameters
        params['barrierType'] = barrier_type
        params['zone_size'] = zone_size

        # Close the setup window
        pygame.quit()
        logging.info("Setup phase complete. Starting simulation.")

        # Launch the main simulation
        simulator.run()
    else:
        # Clean up if user quit without starting simulation
        pygame.quit()
        close_csv_logger(csv_info)
        logging.info("Simulation not started. Exiting.")



def render_help_window(screen, instructions_window, instructions_pos,
                       font, title_font, placing_zone, zone_type,
                       zone_size, directional_percentage, barrier_type):
    """Render the help window with control instructions and current settings"""
    try:
        # Create semi-transparent background
        instructions_window.fill((0, 0, 50))  # Dark blue background

        y_offset = 10

        # Show active zone placement info if applicable
        if placing_zone:
            zone_text = font.render(
                f"Placing {'Safe' if zone_type == 1 else 'Hazard'} Zone | Size: {zone_size}",
                True, (255, 255, 0))
            instructions_window.blit(zone_text, (10, y_offset))

            action_text = font.render("Click to place, C to cancel", True, (255, 255, 0))
            instructions_window.blit(action_text, (10, y_offset + 30))
            y_offset += 60
        else:
            # Display all control instructions
            title = title_font.render("CONTROLS", True, (255, 255, 255))
            instructions_window.blit(title, (10, y_offset))
            y_offset += 30

            # Basic controls
            controls = [
                "SPACE: Start simulation",
                "R: Reset environment",
                "F1: Toggle help display",
                "",
                "Zones:",
                "S: Place safe zone",
                "H: Place hazard zone",
                "C: Cancel zone placement",
                "1/2: Decrease/Increase zone size",
                f"Current zone size: {zone_size}",
                "",
                "Directional Zones:",
                "Arrow Keys: Create directional zones",
                f"SHIFT+1/2: Adjust % ({directional_percentage}%)",
                "",
                "Barriers:",
                "0: No barriers",
                "1: Vertical bar in center",
                "2: Vertical bar in random location",
                "3: Five staggered blocks",
                f"Current barrier type: {barrier_type}",
            ]

            for control in controls:
                if control == "":
                    y_offset += 5  # Less space for separation
                    continue

                control_text = font.render(control, True, (200, 200, 200))
                instructions_window.blit(control_text, (10, y_offset))
                y_offset += 22  # Reduced line spacing

        # Draw the instructions window
        screen.blit(instructions_window, instructions_pos)
    except Exception as e:
        logging.error(f"Error rendering help window: {e}", exc_info=True)


def setup_csv_logger(params):
    """Setup CSV file for logging simulation statistics and evolution metrics"""
    if not params.get('log_to_csv', False):
        return None

    # Ensure log directory exists
    log_folder = params.get('log_folder', 'logs')
    if not os.path.exists(log_folder):
        try:
            os.makedirs(log_folder)
            logging.info(f"Log folder '{log_folder}' created successfully.")
        except Exception as e:
            logging.error(f"Failed to create log folder '{log_folder}': {e}", exc_info=True)
            return None

    try:
        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_folder, f"evolution_log_{timestamp}.csv")

        # Create CSV file with column headers
        file = open(filename, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow([
            'Generation',
            'Population',
            'Creatures_In_Safe_Zone',
            'Safe_Zone_Percentage',
            'Genetic_Diversity',
            'Selection_Method'
        ])

        logging.info(f"Created CSV log file: {filename}")
        return (filename, file, writer)
    except Exception as e:
        logging.error(f"Failed to create CSV log file: {e}", exc_info=True)
        return None


def close_csv_logger(csv_info):
    """Properly close the CSV log file to ensure data is saved"""
    if csv_info and len(csv_info) >= 2:
        filename, file, _ = csv_info
        try:
            file.close()
            logging.info(f"Closed CSV log file: {filename}")
        except Exception as e:
            logging.error(f"Failed to close CSV log file: {e}", exc_info=True)


if __name__ == "__main__":
    main()
