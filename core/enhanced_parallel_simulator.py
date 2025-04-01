import multiprocessing as mp
import numpy as np
import pygame
import logging
import os
import time

from core.simulator import Simulator
from core.simulation_manager import SimulationManager
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, console_level=logging.INFO, file_level=logging.INFO)

class EnhancedParallelSimulator(Simulator):
    """
    Enhanced parallel implementation of the simulator using multiprocessing.
    
    This class extends the base Simulator to parallelize the creature update loop
    using Python's multiprocessing module, similar to OpenMP in the C++ version.
    It uses serializable creatures, shared memory arrays, and a simulation manager
    to coordinate the parallel execution.
    """
    
    def __init__(self, params, grid, signals):
        """
        Initialize the enhanced parallel simulator.
        
        Args:
            params: Simulation parameters dictionary
            grid: The world grid
            signals: Pheromone signal manager
        """
        # Initialize base simulator
        super().__init__(params, grid, signals)
        
        # Create simulation manager
        self.simulation_manager = SimulationManager(params, grid, signals)
        
        # Configure multiprocessing
        self.num_workers = params.get('num_workers', mp.cpu_count())
        self.batch_size = params.get('batch_size', max(1, params['population_size'] // self.num_workers))
        
        logger.info(f"Enhanced parallel simulator initialized with {self.num_workers} workers and batch size {self.batch_size}")
    
    def initialize(self):
        """Initialize the simulation and create the process pool."""
        # Reset grid
        self.grid.reset()
        
        # Create barriers
        self.barrier_manager.create_barriers(self.params.get('barrierType', 0))
        
        # Don't create random zones - this is causing the red and green blocks
        # self.zone_manager.place_random_zones()
        
        # Setup radiation if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.setup_radiation()
        
        # Initialize population
        self.population.initialize(self.grid)
        
        # Reset counters
        self.generation = 0
        self.step = 0
        self.murder_count = 0
        
        # Initialize simulation manager
        self.simulation_manager.zone_manager = self.zone_manager
        self.simulation_manager.barrier_manager = self.barrier_manager
        self.simulation_manager.radiation_manager = self.radiation_manager
        self.simulation_manager.set_population(self.population)
        if hasattr(self, 'logger'):
            self.simulation_manager.set_logger(self.logger)
        
        # Initialize the simulation manager
        self.simulation_manager.initialize()
        
        logger.info("Enhanced parallel simulator initialized")
    
    def update(self):
        """
        Update one simulation step using parallel processing.
        
        This overrides the base update method to parallelize the creature update loop.
        """
        if self.paused:
            return
        
        # Use simulation manager to update creatures in parallel
        self.simulation_manager.update()
        
        # Update simulation state from simulation manager
        self.step = self.simulation_manager.step
        self.generation = self.simulation_manager.generation
        self.murder_count = self.simulation_manager.murder_count
        
        # Process challenge-specific effects
        self._handle_challenge_specific_effects()
    
    def end_generation(self):
        """
        Handle end of generation logic.
        
        This is handled by the simulation manager, so we don't need to do anything here.
        """
        pass
    
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
                pass
        
        # Delete other log files
        log_files = glob.glob("*.log") + glob.glob("*/*.log")
        for file in log_files:
            try:
                os.remove(file)
                logger.info(f"Deleted log file: {file}")
            except Exception as e:
                logger.error(f"Failed to delete log file {file}: {e}")
                pass
    
    def run(self):
        """Run the main simulation loop with proper cleanup."""
        # Initialize before starting
        self.initialize()
        
        # Ensure pygame is initialized and renderer is set up
        if not pygame.get_init():
            logger.info("Initializing pygame in EnhancedParallelSimulator.run()")
            pygame.init()
        
        # Reinitialize the renderer
        from visualization.renderer import Renderer
        self.renderer = Renderer(self.params)
        logger.info("Renderer reinitialized")
        
        try:
            # Main simulation loop
            while self.running:
                # Handle events
                try:
                    self.running, self.paused, key_events = self.renderer.handle_events()
                except ValueError:
                    # Handle case where handle_events returns only running and paused
                    self.running, self.paused = self.renderer.handle_events()
                    key_events = []
                
                # Check for special key events
                for event in key_events:
                    # Force new generation when G is pressed and simulation is paused
                    if event.key == pygame.K_g and self.paused:
                        self.step = self.params['steps_per_generation']  # This will trigger a new generation
                        self.simulation_manager.step = self.step
                        logger.info("Forced new generation")
                    
                    # Reset simulation when R is pressed and simulation is paused
                    elif event.key == pygame.K_r and self.paused:
                        # Delete log files from previous run
                        self._delete_log_files()
                        # Re-initialize simulation
                        self.initialize()
                        logger.info("Simulation reset and log files deleted")
                    
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
                
                # Update simulation
                if not self.paused:
                    start_time = time.time()
                    self.update()
                    end_time = time.time()
                    update_time = end_time - start_time
                    
                    # Log update time every 10 steps
                    if self.step % 10 == 0:
                        logger.debug(f"Update time: {update_time:.6f} seconds")
                
                # Update renderer with current state
                self.renderer.set_generation(self.generation)
                self.renderer.set_step(self.step)
                
                # Update params with current step for challenge visualization
                self.params['current_step'] = self.step
                
                self.renderer.render_world(self.grid, self.population.creatures)
        
        finally:
            # Clean up when simulation ends
            self.simulation_manager.close()
            self.logger.close()
            
            # Only quit pygame if it's initialized
            if pygame.get_init():
                pygame.quit()
