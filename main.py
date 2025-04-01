import os
import argparse
import pygame
import logging
import datetime
import csv
import multiprocessing

from utils.logging_config import configure_logging, get_logger, set_logging_level
from core.params import load_parameters
from core.grid import Grid
from core.signals import Signals
from visualization.interactive_simulator import InteractiveSimulator
from core.parallel_simulator import ParallelSimulator
from core.enhanced_parallel_simulator import EnhancedParallelSimulator
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from visualization.renderer import GridRenderer

# Configure default logging
configure_logging()
logger = get_logger(__name__)
logging.getLogger('pygame').setLevel(logging.CRITICAL)  # Reduce pygame log verbosity


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
        # Choose simulator based on configuration
        if params.get('enable_parallelization', False):
            if params.get('use_enhanced_parallelization', False):
                logging.info("Using enhanced parallel simulator with multiprocessing")
                simulator = EnhancedParallelSimulator(params, grid, signals)
            else:
                logging.info("Using parallel simulator with multiprocessing")
                simulator = ParallelSimulator(params, grid, signals)
        else:
            simulator = InteractiveSimulator(params, grid, signals)
            
        simulator.zone_manager = zone_manager
        simulator.barrier_manager = barrier_manager
        grid.zone_manager = zone_manager  # For reset operations
        logging.info("Simulator initialized with ZoneManager and BarrierManager.")
    except Exception as e:
        logging.error("Failed to initialize Simulator", exc_info=True)
        print(f"Failed to initialize Simulator: {e}")
        pygame.quit()  # Clean up pygame if simulator initialization fails
        return

    # Set default values for parameters that would normally be configured during setup
    zone_size = params.get('zone_size', 50)  # Default zone diameter
    barrier_type = params.get('barrierType', 0)
    
    # Apply barrier type
    barrier_manager.create_barriers(barrier_type)
    
    # Handle challenge-specific zone setup
    challenge_type = params.get('challenge', 0)
    
    # Clear any existing zones first
    zone_manager.clear_zones()
    
    # Only create actual zones for specific challenge types
    # For most challenges, we'll rely on the challenge highlighting and creature detection
    if challenge_type == 9:  # CHALLENGE_LEFT_EIGHTH
        zone_manager.create_directional_zone('left', 12.5, 1)  # 12.5% of the map, safe zone (type 1)
    elif challenge_type in [0, 1, 2, 4, 5, 6, 8, 13, 17]:
        # These challenges are handled by challenge highlighting and creature detection:
        # CHALLENGE_CIRCLE, CHALLENGE_RIGHT_HALF, CHALLENGE_RIGHT_QUARTER,
        # CHALLENGE_CENTER_WEIGHTED, CHALLENGE_CENTER_UNWEIGHTED, CHALLENGE_CENTER_SPARSE,
        # CHALLENGE_CORNER, CHALLENGE_CORNER_WEIGHTED, CHALLENGE_EAST_WEST_EIGHTHS,
        # CHALLENGE_ALTRUISM
        pass
    
    # CSV logging is handled by the Logger class in the simulator
    
    # Save parameters
    params['zone_size'] = zone_size
    params['barrierType'] = barrier_type
    
    # Close the pygame window
    pygame.quit()
    logging.info("Starting simulation directly (setup phase skipped).")
    
    # Launch the main simulation
    simulator.run()



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
