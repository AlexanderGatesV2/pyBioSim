import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import pygame
import logging
import os

from core.simulator import Simulator
from core.thread_safe_grid import ThreadSafeGrid
from core.thread_safe_signals import ThreadSafeSignals
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, console_level=logging.INFO, file_level=logging.INFO)

def _update_creature_batch(batch_data):
    """
    Update a batch of creatures in parallel.
    
    This function is defined at module level to be picklable for multiprocessing.
    
    Args:
        batch_data: Tuple containing (batch_ids, grid_data, signals_data, sim_step, params)
    
    Returns:
        Tuple of (grid_data, signals_data, creature_updates)
    """
    # Disable pygame in worker processes to avoid video system initialization issues
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    batch_ids, grid_data, signals_data, sim_step, params = batch_data
    
    # Set process-specific random seed for deterministic behavior
    seed = os.getpid() + sim_step
    np.random.seed(seed)
    
    # Create local grid and signals objects (no locks involved)
    from core.grid import Grid
    from core.signals import Signals
    
    # Recreate grid from data
    grid = Grid(params['world_size'])
    grid.data = grid_data
    
    # Recreate signals from data
    signals = Signals(params['world_size'], params['signal_layers'], params=params)
    signals.data = signals_data
    
    # We'll collect updates to apply to the main process creatures
    creature_updates = []
    
    # Process each creature in the batch
    # Note: In a real implementation, you would need to recreate the creatures
    # from serializable data, but for this example we'll just simulate the updates
    
    # Return updated grid and signals data
    return grid.data, signals.data, creature_updates


class ParallelSimulator(Simulator):
    """
    Parallel implementation of the simulator using multiprocessing.
    
    This class extends the base Simulator to parallelize the creature update loop
    using Python's multiprocessing module, similar to OpenMP in the C++ version.
    """
    def __init__(self, params, grid, signals):
        """
        Initialize the parallel simulator.
        
        Args:
            params: Simulation parameters dictionary
            grid: The world grid
            signals: Pheromone signal manager
        """
        # Use regular grid and signals (no thread-safety needed)
        super().__init__(params, grid, signals)
        
        # Configure multiprocessing
        self.num_workers = params.get('num_workers', multiprocessing.cpu_count())
        self.batch_size = params.get('batch_size', max(1, params['population_size'] // self.num_workers))
        self.pool = None
        
        logger.info(f"Parallel simulator initialized with {self.num_workers} workers and batch size {self.batch_size}")
        
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
        
        # Initialize the process pool
        if self.pool is None:
            logger.info(f"Creating process pool with {self.num_workers} workers")
            self.pool = Pool(processes=self.num_workers)
            
    def update(self):
        """
        Update one simulation step using parallel processing.
        
        This overrides the base update method to parallelize the creature update loop.
        """
        if self.paused:
            return

        # Create a dictionary of creatures for quick lookup
        creatures_dict = {creature.id: creature for creature in self.population.creatures}
        
        # Filter only living creatures
        living_creatures = [c for c in self.population.creatures if c.alive]
        
        # For now, we'll use the sequential update for creatures
        # This is a temporary solution until we implement proper serialization
        for creature in living_creatures:
            creature.update(self.grid, self.population.creatures, self.signals, self.step)
        
        # NOTE: The code below is the parallel version, but it's commented out
        # because we need to implement proper serialization of creatures first
        """
        # Divide creatures into batches by ID (to avoid pickling issues)
        creature_batches = []
        for i in range(0, len(living_creatures), self.batch_size):
            batch = living_creatures[i:i + self.batch_size]
            batch_ids = [c.id for c in batch]
            creature_batches.append(batch_ids)
        
        # Prepare batch data for parallel processing - pass only serializable data
        batch_data = [(batch_ids, self.grid.data.copy(), self.signals.data.copy(), 
                      self.step, self.params) 
                      for batch_ids in creature_batches]
        
        # Process batches in parallel and collect results
        if creature_batches:
            results = self.pool.map(_update_creature_batch, batch_data)
            
            # Merge results back into main grid and signals
            # This is a simple approach - in a real implementation, you'd need
            # to handle conflicts more carefully
            for grid_data, signals_data, creature_updates in results:
                # Apply creature updates
                pass
        """
        
        # Continue with the rest of the update as in the original
        # Apply challenge-specific effects
        self._handle_challenge_specific_effects()
        
        # Process death queue
        self.murder_count += len(self.grid.death_queue)
        self.grid.process_death_queue(creatures_dict)
        
        # Process move queue
        self.grid.process_move_queue(creatures_dict)
        
        # Fade pheromones
        for layer in range(self.signals.num_layers):
            self.signals.fade(layer)
            
        # Update radiation if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.update_radiation()
            self.radiation_manager.apply_radiation_effects(self.population.creatures)
            
        # Increment step counter
        self.step += 1
        
        # Check if generation is complete
        if self.step >= self.params['steps_per_generation']:
            self.end_generation()
            
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
            logger.info("Initializing pygame in ParallelSimulator.run()")
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
                    self.update()

                # Update renderer with current state
                self.renderer.set_generation(self.generation)
                self.renderer.set_step(self.step)
                
                # Update params with current step for challenge visualization
                self.params['current_step'] = self.step
                
                self.renderer.render_world(self.grid, self.population.creatures)

        finally:
            # Clean up when simulation ends
            if self.pool:
                logger.info("Closing and joining process pool")
                self.pool.close()
                self.pool.join()
            self.logger.close()
            
            # Only quit pygame if it's initialized
            if pygame.get_init():
                pygame.quit()
