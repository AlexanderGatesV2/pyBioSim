import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from queue import Empty as QueueEmpty
import numpy as np
import logging
import time
import os
import random
import math
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='simulation_manager_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SimulationManager:
    """
    Manages the parallel execution of the simulation.
    
    This class coordinates the creation and management of worker processes,
    distributes work among them, and synchronizes the results.
    """
    
    def __init__(self, params, grid, signals):
        """
        Initialize the simulation manager.
        
        Args:
            params: Simulation parameters
            grid: The world grid
            signals: Pheromone signal manager
        """
        self.params = params
        self.grid = grid
        self.signals = signals
        
        # Configure multiprocessing
        self.num_workers = params.get('num_workers', mp.cpu_count())
        self.batch_size = params.get('batch_size', max(1, params['population_size'] // self.num_workers))
        
        # Initialize worker processes and queues
        self.workers = []
        self.task_queues = []
        self.result_queues = []
        self.stop_events = []
        
        # Initialize simulation state
        self.generation = 0
        self.step = 0
        self.running = True
        self.paused = False
        
        # Initialize creature cache
        self.creatures = []
        self.serializable_creatures = []
        
        # Initialize zone and barrier managers (will be set by simulator)
        self.zone_manager = None
        self.barrier_manager = None
        self.radiation_manager = None
        
        # Initialize death and move queues
        self.death_queue = []
        self.move_queue = []
        
        # Initialize murder count
        self.murder_count = 0
        
        logger.info(f"SimulationManager initialized with {self.num_workers} workers and batch size {self.batch_size}")
    
    def initialize(self):
        """
        Initialize the simulation manager.
        """
        # Convert grid and signals to shared memory versions
        from core.shared_memory_grid import SharedMemoryGrid
        from core.shared_memory_signals import SharedMemorySignals
        
        # Create shared memory grid and signals
        self.shared_grid = SharedMemoryGrid.from_grid(self.grid)
        self.shared_signals = SharedMemorySignals.from_signals(self.signals)
        
        # Get shared memory names
        self.grid_shm_name = self.shared_grid.get_shm_name()
        self.signals_shm_name = self.shared_signals.get_shm_name()
        
        # Initialize worker processes and queues
        for i in range(self.num_workers):
            # Create task and result queues
            task_queue = Queue()
            result_queue = Queue()
            stop_event = Event()
            
            # Create worker process
            worker = Process(
                target=self._worker_process,
                args=(i, task_queue, result_queue, stop_event, self.grid_shm_name, self.signals_shm_name, self.params),
                daemon=True
            )
            
            # Store worker and queues
            self.workers.append(worker)
            self.task_queues.append(task_queue)
            self.result_queues.append(result_queue)
            self.stop_events.append(stop_event)
            
            # Start worker process
            worker.start()
            
        logger.info(f"Started {self.num_workers} worker processes")
    
    def close(self):
        """
        Close the simulation manager and clean up resources.
        """
        # Signal workers to stop
        for stop_event in self.stop_events:
            stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
        
        # Close shared memory
        self.shared_grid.close()
        self.shared_signals.close()
        
        logger.info("SimulationManager closed")
    
    @staticmethod
    def _worker_process(worker_id, task_queue, result_queue, stop_event, grid_shm_name, signals_shm_name, params):
        """
        Worker process function.
        
        Args:
            worker_id: ID of the worker process
            task_queue: Queue for receiving tasks
            result_queue: Queue for sending results
            stop_event: Event for signaling the worker to stop
            grid_shm_name: Name of the shared memory block for the grid
            signals_shm_name: Name of the shared memory block for the signals
            params: Simulation parameters
        """
        try:
            # Set process-specific random seed
            seed = os.getpid() + worker_id
            random.seed(seed)
            np.random.seed(seed)
            
            # Import required modules
            from core.shared_memory_grid import SharedMemoryGrid
            from core.shared_memory_signals import SharedMemorySignals
            from agents.serializable_creature import SerializableCreature
            
            # Attach to shared memory grid and signals
            grid = SharedMemoryGrid(params['world_size'], existing_shm_name=grid_shm_name)
            signals = SharedMemorySignals(params['world_size'], params['signal_layers'], params, existing_shm_name=signals_shm_name)
            
            logger.info(f"Worker {worker_id} initialized")
            
            # Main worker loop
            while not stop_event.is_set():
                try:
                    # Get task from queue with timeout
                    task = task_queue.get(timeout=0.1)
                    
                    # Process task
                    if task['type'] == 'update_creatures':
                        # Get task data
                        serializable_creatures = task['creatures']
                        sim_step = task['step']
                        
                        # Update creatures
                        updated_creatures, death_queue, move_queue = SimulationManager._update_creatures(
                            serializable_creatures, grid, signals, sim_step, params
                        )
                        
                        # Send results
                        result_queue.put({
                            'type': 'update_creatures',
                            'creatures': updated_creatures,
                            'death_queue': death_queue,
                            'move_queue': move_queue
                        })
                    
                    elif task['type'] == 'shutdown':
                        # Exit worker loop
                        break
                    
                except QueueEmpty:
                    # No task available, continue
                    pass
                
                except Exception as e:
                    # Log error and continue
                    logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                    
                    # Send error result
                    result_queue.put({
                        'type': 'error',
                        'error': str(e)
                    })
            
            # Clean up
            grid.close()
            signals.close()
            
            logger.info(f"Worker {worker_id} stopped")
        
        except Exception as e:
            # Log error
            logger.error(f"Worker {worker_id} initialization error: {e}", exc_info=True)
    
    @staticmethod
    def _update_creatures(serializable_creatures, grid, signals, sim_step, params):
        """
        Update a batch of creatures.
        
        Args:
            serializable_creatures: List of serializable creatures to update
            grid: Shared memory grid
            signals: Shared memory signals
            sim_step: Current simulation step
            params: Simulation parameters
            
        Returns:
            Tuple of (updated_creatures, death_queue, move_queue)
        """
        # Import SerializableCreature here to avoid circular imports
        from agents.serializable_creature import SerializableCreature
        # Convert serializable creatures to regular creatures
        creatures = []
        for serializable_creature in serializable_creatures:
            creature = serializable_creature.to_creature(params)
            # Ensure last_move_offset is properly initialized
            if not hasattr(creature, 'last_move_offset') or creature.last_move_offset is None:
                creature.last_move_offset = (0, 0)
            creatures.append(creature)
        
        # Create a dictionary of creatures for quick lookup
        creatures_dict = {creature.id: creature for creature in creatures}
        
        # Initialize death and move queues
        death_queue = []
        move_queue = []
        
        # Update each creature
        for creature in creatures:
            if creature.alive:
                # Update creature
                logger.info(f"Updating creature {creature.id} at position {creature.position}")
                creature.update(grid, creatures, signals, sim_step)
                logger.info(f"After update, creature {creature.id} is at position {creature.position}")
                
                # Check if creature queued for death
                if not creature.alive:
                    death_queue.append(creature.id)
                
                # Check if creature moved
                # In the original implementation, the creature would queue its move in the grid
                # Here, we need to check if the creature's position changed and queue the move ourselves
                
                # Find the corresponding serializable creature in the input list
                for sc in serializable_creatures:
                    if sc.id == creature.id:
                        old_x, old_y = int(sc.position[0]), int(sc.position[1])
                        new_x, new_y = int(creature.position[0]), int(creature.position[1])
                        
                        if (old_x, old_y) != (new_x, new_y):
                            move_queue.append((creature.id, (new_x, new_y)))
                            # Update last_move_offset based on the actual movement
                            creature.last_move_offset = (new_x - old_x, new_y - old_y)
                            logger.info(f"Queued move for creature {creature.id} from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                        break
        
        # Convert updated creatures back to serializable creatures
        updated_serializable_creatures = []
        for creature in creatures:
            serializable_creature = SerializableCreature()
            serializable_creature.from_creature(creature)
            updated_serializable_creatures.append(serializable_creature)
        
        return updated_serializable_creatures, death_queue, move_queue
    
    def update(self):
        """
        Update one simulation step using parallel processing.
        """
        if self.paused:
            return
        
        # Convert creatures to serializable creatures
        if not self.serializable_creatures:
            from agents.serializable_creature import SerializableCreature
            
            self.serializable_creatures = []
            for creature in self.creatures:
                serializable_creature = SerializableCreature()
                serializable_creature.from_creature(creature)
                self.serializable_creatures.append(serializable_creature)
        
        # Divide creatures into batches
        batches = []
        for i in range(0, len(self.serializable_creatures), self.batch_size):
            batch = self.serializable_creatures[i:i + self.batch_size]
            batches.append(batch)
        
        # Ensure we have at least one batch per worker
        while len(batches) < self.num_workers:
            batches.append([])
        
        # Distribute batches to workers
        for i, batch in enumerate(batches):
            worker_id = i % self.num_workers
            
            # Send task to worker
            self.task_queues[worker_id].put({
                'type': 'update_creatures',
                'creatures': batch,
                'step': self.step
            })
        
        # Collect results from workers
        updated_creatures = []
        death_queue = []
        move_queue = []
        
        for i in range(len(batches)):
            worker_id = i % self.num_workers
            
            # Get result from worker
            result = self.result_queues[worker_id].get()
            
            if result['type'] == 'update_creatures':
                # Add updated creatures to list
                updated_creatures.extend(result['creatures'])
                
                # Add death and move queues
                death_queue.extend(result['death_queue'])
                move_queue.extend(result['move_queue'])
            
            elif result['type'] == 'error':
                # Log error
                logger.error(f"Worker error: {result['error']}")
        
        # Update serializable creatures
        self.serializable_creatures = updated_creatures
        
        # Update death and move queues
        self.death_queue.extend(death_queue)
        self.move_queue.extend(move_queue)
        
        # Process death queue
        creatures_dict = {creature.id: creature for creature in self.creatures}
        self.murder_count += len(self.death_queue)
        
        for creature_id in self.death_queue:
            if creature_id in creatures_dict:
                creature = creatures_dict[creature_id]
                x, y = int(creature.position[0]), int(creature.position[1])
                
                # Remove creature from grid
                if 0 <= x < self.grid.size[0] and 0 <= y < self.grid.size[1]:
                    if self.grid.data[x, y, 0] == creature_id:
                        self.grid.data[x, y, 0] = 0
                
                # Mark creature as dead
                creature.alive = False
        
        # Clear death queue
        self.death_queue = []
        
        # Process move queue
        logger.info(f"Processing {len(self.move_queue) + len(self.shared_grid.move_queue)} moves")
        
        # Process moves from the simulation manager's move queue
        for creature_id, new_pos in self.move_queue:
            if creature_id in creatures_dict:
                creature = creatures_dict[creature_id]
                old_x, old_y = int(creature.position[0]), int(creature.position[1])
                new_x, new_y = int(new_pos[0]), int(new_pos[1])
                
                logger.info(f"Processing move for creature {creature_id} from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                
                # Check if the new position is valid
                if (0 <= new_x < self.grid.size[0] and 
                    0 <= new_y < self.grid.size[1] and 
                    not self.grid.is_barrier_at(new_x, new_y)):
                    
                    # Check if the new position is empty
                    if self.grid.data[new_x, new_y, 0] == 0:
                        # Remove creature from old position
                        if 0 <= old_x < self.grid.size[0] and 0 <= old_y < self.grid.size[1]:
                            if self.grid.data[old_x, old_y, 0] == creature_id:
                                self.grid.data[old_x, old_y, 0] = 0
                        
                        # Add creature to new position
                        self.grid.data[new_x, new_y, 0] = creature_id
                        
                        # Update creature position
                        creature.position = (new_x, new_y)
                        
                        # Calculate last move offset and direction change
                        dx = new_x - old_x
                        dy = new_y - old_y
                        creature.last_move_offset = (dx, dy)
                        
                        # Update direction based on movement
                        if dx != 0 or dy != 0:
                            magnitude = math.sqrt(dx * dx + dy * dy)
                            if magnitude > 0:
                                creature.direction = (dx / magnitude, dy / magnitude)
                        
                        logger.info(f"Moved creature {creature_id} to ({new_x}, {new_y})")
                    else:
                        logger.info(f"Cannot move creature {creature_id} to ({new_x}, {new_y}) - position occupied")
                else:
                    logger.info(f"Cannot move creature {creature_id} to ({new_x}, {new_y}) - invalid position")
        
        # Process moves from the shared grid's move queue
        with self.shared_grid.move_queue_lock:
            # Print the shared grid's move queue for debugging
            logger.info(f"Shared grid move queue: {self.shared_grid.move_queue}")
            for creature_id, new_pos in self.shared_grid.move_queue:
                if creature_id in creatures_dict:
                    creature = creatures_dict[creature_id]
                    old_x, old_y = int(creature.position[0]), int(creature.position[1])
                    new_x, new_y = int(new_pos[0]), int(new_pos[1])
                    
                    logger.info(f"Processing move from shared grid for creature {creature_id} from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                    
                    # Check if the new position is valid
                    if (0 <= new_x < self.grid.size[0] and 
                        0 <= new_y < self.grid.size[1] and 
                        not self.grid.is_barrier_at(new_x, new_y)):
                        
                        # Check if the new position is empty
                        if self.grid.data[new_x, new_y, 0] == 0:
                            # Remove creature from old position
                            if 0 <= old_x < self.grid.size[0] and 0 <= old_y < self.grid.size[1]:
                                if self.grid.data[old_x, old_y, 0] == creature_id:
                                    self.grid.data[old_x, old_y, 0] = 0
                            
                            # Add creature to new position
                            self.grid.data[new_x, new_y, 0] = creature_id
                            
                            # Update creature position
                            creature.position = (new_x, new_y)
                            
                            # Calculate last move offset and direction change
                            dx = new_x - old_x
                            dy = new_y - old_y
                            creature.last_move_offset = (dx, dy)
                            
                            # Update direction based on movement
                            if dx != 0 or dy != 0:
                                magnitude = math.sqrt(dx * dx + dy * dy)
                                if magnitude > 0:
                                    creature.direction = (dx / magnitude, dy / magnitude)
                            
                            logger.info(f"Moved creature {creature_id} to ({new_x}, {new_y})")
                        else:
                            logger.info(f"Cannot move creature {creature_id} to ({new_x}, {new_y}) - position occupied")
                    else:
                        logger.info(f"Cannot move creature {creature_id} to ({new_x}, {new_y}) - invalid position")
            
            # Clear shared grid's move queue
            self.shared_grid.move_queue = []
        
        # Clear simulation manager's move queue
        self.move_queue = []
        
        # Update radiation if enabled
        if self.params.get('enable_radioactive_environment', False) and self.radiation_manager is not None:
            self.radiation_manager.update_radiation()
            self.radiation_manager.apply_radiation_effects(self.creatures)
        
        # Fade pheromones
        for layer in range(self.signals.num_layers):
            self.signals.fade(layer)
        
        # Increment step counter
        self.step += 1
        
        # Check if generation is complete
        if self.step >= self.params['steps_per_generation']:
            self.end_generation()
    
    def end_generation(self):
        """
        Handle end of generation logic.
        """
        # Log generation stats
        if hasattr(self, 'logger'):
            self.logger.log_generation(self.generation, self.creatures, self.murder_count)
        
        # Perform natural selection
        self.creatures = self.population.natural_selection_tournament(self.grid)
        
        # Increment generation counter and reset step counter
        self.generation += 1
        self.step = 0
        self.murder_count = 0
        
        # Ultra-fast clearing using numpy operations and the non_barrier_mask
        creature_layer = self.grid.data[:, :, 0]
        creature_layer[self.grid.non_barrier_mask] = 0
        
        # Place creatures in valid locations
        empty_positions = []
        
        # Pre-generate a pool of empty positions
        for _ in range(min(len(self.creatures) * 2, 1000)):
            try:
                pos = self.grid.find_empty_location()
                empty_positions.append(pos)
            except Exception:
                break
        
        # Place creatures using the pre-generated positions
        for i, creature in enumerate(self.creatures):
            if i < len(empty_positions):
                position = empty_positions[i]
            else:
                position = self.grid.find_empty_location()
            
            # Set creature position and update grid
            creature.position = position
            self.grid.data[int(position[0]), int(position[1]), 0] = creature.id
            
            # Initialize creature's last_move_offset to (0, 0)
            creature.last_move_offset = (0, 0)
            
            # Ensure direction is properly initialized
            if not hasattr(creature, 'direction') or creature.direction == (0, 0):
                angle = random.uniform(0, 2 * math.pi)
                creature.direction = (math.cos(angle), math.sin(angle))
        
        # Rebuild serializable creatures cache with the new creatures
        # This must be done AFTER placing creatures to ensure the serializable creatures
        # have the correct position and movement information
        self.serializable_creatures = []
        from agents.serializable_creature import SerializableCreature
        for creature in self.creatures:
            serializable_creature = SerializableCreature()
            serializable_creature.from_creature(creature)
            self.serializable_creatures.append(serializable_creature)
    
    def set_population(self, population):
        """
        Set the population for the simulation manager.
        
        Args:
            population: Population object
        """
        self.population = population
        self.creatures = population.creatures
    
    def set_logger(self, logger):
        """
        Set the logger for the simulation manager.
        
        Args:
            logger: Logger object
        """
        self.logger = logger
