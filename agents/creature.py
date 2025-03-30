import numpy as np
import logging
import random
import math

from core.types import Action, Sensor, calculate_genetic_similarity
from agents.neural_network import NeuralNetwork
from agents.genome import Genome
from utils.random_generator import random_generator
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='creature_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class Creature:
    next_id = 1  # Class variable for unique IDs

    def __init__(self, genome=None, position=None, params=None):
        """
        Initialize a creature with a genome and position

        Args:
            genome: The creature's genetic code (optional, random if None)
            position: Starting position (optional, random if None)
            params: Simulation parameters
        """
        self.id = Creature.next_id
        Creature.next_id += 1

        self.genome = genome if genome else Genome(length=params['genome_length'])
        self.params = params
        self.alive = True

        # Ensure starting position is within bounds
        if position:
            self.position = (min(params['world_size'][0] - 1, max(0, position[0])),
                             min(params['world_size'][1] - 1, max(0, position[1])))
        else:
            self.position = (random.randint(2, params['world_size'][0] - 3),
                             random.randint(2, params['world_size'][1] - 3))

        # Initialize direction with random uniform values using our generator
        self.direction = (random_generator.random_float() * 2 - 1,
                          random_generator.random_float() * 2 - 1)

        # Calculate the magnitude of the direction vector
        magnitude = math.sqrt(self.direction[0] ** 2 + self.direction[1] ** 2)
        # Normalize the direction vector if the magnitude is greater than 0
        if magnitude > 0:
            self.direction = (self.direction[0] / magnitude, self.direction[1] / magnitude)
        else:
            # If the magnitude is 0, set to a random non-zero direction
            angle = random.uniform(0, 2 * math.pi)  # Choose a random angle
            self.direction = (math.cos(angle), math.sin(angle))

        # Creature state variables
        self.age = 0
        self.energy = 1000  # Default energy value
        self.in_safe_zone = False
        self.has_killed = False

        # Enhanced state tracking for more complex behaviors
        self.birth_position = self.position  # Track where creature was born
        self.last_positions = []  # Track recent positions (for detecting if stuck)
        self.max_position_history = 10
        self.stuck_counter = 0  # Count steps where creature hasn't moved

        # Metabolic and health factors
        self.health = 1.0  # Health factor (1.0 = perfect health)
        self.metabolic_rate = 0.1  # Base energy consumption per step
        self.energy_efficiency = 1.0  # Multiplier for energy consumption (lower = more efficient)

        # Behavioral traits using our random generator
        self.aggression = random_generator.random_float()
        self.curiosity = random_generator.random_float()
        self.sociality = random_generator.random_float()

        # Enhanced sensory capabilities
        self.longprobe_dist = params.get('longProbeDistance', 16)
        self.recent_signals = []  # Track recently detected signals
        self.detected_creatures = []  # Track recently detected creatures

        # Neural network characteristics
        self.responsiveness = 0.5  # Affects how strongly actions are executed (0.0-1.0)
        self.oscPeriod = 34  # Period for oscillator sensor

        # Initialize the neural network (brain)
        self.brain = NeuralNetwork(self.genome, params)
        
        # Store last discrete movement offset (matches C++ lastMoveDir concept)
        self.last_move_offset = (0, 0) # Initialize to no movement

    def apply_movement(self, movement_info, grid):
        """
        Apply movement to the creature and update the grid
        
        Args:
            movement_info: Dictionary with movement data from neural network
            grid: The world grid
            
        Returns:
            bool: True if movement was successful, False otherwise
        """
        dx, dy = movement_info['discrete']
        
        if dx == 0 and dy == 0:
            return False  # No movement
        
        # Apply boundary checks
        new_x = max(0, min(self.params['world_size'][0] - 1, self.position[0] + dx))
        new_y = max(0, min(self.params['world_size'][1] - 1, self.position[1] + dy))
        
        # Check if the target position is a barrier
        if grid.is_barrier_at(int(new_x), int(new_y)):
            # Cannot move into a barrier
            return False
        
        # Update direction if provided
        if movement_info['direction_update']:
            self.direction = movement_info['direction_update']
        
        # Queue the movement for processing at the end of the step
        # This prevents race conditions where multiple creatures try to move to the same cell
        grid.queue_for_move(self.id, (new_x, new_y))
        
        # We'll consider this a successful movement attempt, even though
        # it might be rejected later if the destination is occupied
        # Update stuck detection
        self.last_positions.append(self.position)
        if len(self.last_positions) > self.max_position_history:
            self.last_positions.pop(0)
        
        # Reset stuck counter since we attempted to move
        self.stuck_counter = 0
        return True
            
    def _process_zone_effects(self, zone_type):
        """Process effects of being in different zone types"""
        if zone_type == 1:  # Safe zone
            self.in_safe_zone = True
            # Add energy bonus for being in safe zone
            self.energy += self.params.get('safe_zone_bonus', 10.0)
        elif zone_type == 2:  # Hazard zone
            self.in_safe_zone = False
            # Apply energy penalty for being in hazard zone
            self.energy -= self.params.get('hazard_zone_penalty', 10.0)
            self.energy = max(0.0, self.energy)  # Prevent negative energy
        else:
            self.in_safe_zone = False

    def _apply_state_updates(self, state_updates, signals, grid=None, creatures=None):
        """Apply non-movement state updates from neural network"""
        for key, value in state_updates.items():
            if key == 'emit_signal0' and value and signals is not None:
                # Handle signal emission
                signals.increment(0, int(self.position[0]), int(self.position[1]))
            elif key == 'attempt_kill' and value and grid is not None and creatures is not None:
                # Handle kill attempt
                self._attempt_kill(grid, creatures)
            else:
                # Update creature state variables
                setattr(self, key, value)
                
    def update(self, grid, creatures, signals):
        """Update the creature for one simulation step"""
        if not self.alive:
            return

        # Record current position and increment age
        self.age += 1

        # Check for zone effects at current position
        x, y = int(self.position[0]), int(self.position[1])
        zone_type = grid.data[x, y, 2]  # Get zone type from grid

        # Update creature's zone status and apply zone effects
        self._process_zone_effects(zone_type)
        
        # Get sensory inputs based on current environment, passing simStep
        # We need the global simStep, which is managed by the Simulator class
        # Let's assume it's passed via params for now, or we modify update signature later
        sim_step = self.params.get('current_step', self.age) # Use age as fallback if not passed
        sensory_inputs = self.get_sensory_inputs(grid, creatures, signals, sim_step)
        
        # Create state dictionary to pass creature context to neural network
        creature_state = {
            'direction': self.direction,
            'responsiveness': self.responsiveness,
            'oscPeriod': self.oscPeriod,
            'longprobe_dist': self.longprobe_dist
        }
        
        # Process neural network with enhanced state interaction
        action_values, state_updates = self.brain.feed_forward(
            sensory_inputs, self.age, creature_state)
        
        # Apply neural network state updates
        self._apply_state_updates(state_updates, signals, grid, creatures)
        
        # Process movement using the consolidated logic
        movement_info = self.brain.process_movement(action_values, creature_state)
        self.apply_movement(movement_info, grid)
        
        # Update health and detect environment
        self.update_health()
        self.detect_environment(grid, creatures)

    def get_sensory_inputs(self, grid, creatures, signals, sim_step):
        """
        Get values for all sensory neurons based on current environment

        Args:
            grid: The world grid
            creatures: List of all creatures
            signals: Signal manager
            sim_step: Current simulation step number

        Returns:
            np.array of sensor values matching sensor indices
        """
        # Create array for all possible sensors
        sensory_values = np.zeros(self.params['num_sensory_neurons'])

        # Creature position and last discrete move
        x, y = self.position
        last_dx, last_dy = self.last_move_offset # Use discrete offset for sensors

        # Fill sensory values for each sensor type if it exists in our configuration

        # Position sensors (match C++ normalization)
        if Sensor.LOC_X.value < len(sensory_values):
            sensory_values[Sensor.LOC_X.value] = x / (self.params['world_size'][0] - 1.0)

        if Sensor.LOC_Y.value < len(sensory_values):
            sensory_values[Sensor.LOC_Y.value] = y / (self.params['world_size'][1] - 1.0)

        # Boundary distance sensors
        if Sensor.BOUNDARY_DIST_X.value < len(sensory_values):
            dist_x_min = min(x, self.params['world_size'][0] - 1 - x)
            max_x = self.params['world_size'][0] / 2
            sensory_values[Sensor.BOUNDARY_DIST_X.value] = dist_x_min / max_x

        if Sensor.BOUNDARY_DIST_Y.value < len(sensory_values):
            dist_y_min = min(y, self.params['world_size'][1] - 1 - y)
            max_y = self.params['world_size'][1] / 2
            sensory_values[Sensor.BOUNDARY_DIST_Y.value] = dist_y_min / max_y

        if Sensor.BOUNDARY_DIST.value < len(sensory_values):
            dist_x_min = min(x, self.params['world_size'][0] - 1 - x)
            dist_y_min = min(y, self.params['world_size'][1] - 1 - y)
            closest = min(dist_x_min, dist_y_min)
            max_possible = max(self.params['world_size'][0] / 2 - 1,
                               self.params['world_size'][1] / 2 - 1)
            sensory_values[Sensor.BOUNDARY_DIST.value] = closest / max_possible

        # Last movement direction (use discrete offset)
        if Sensor.LAST_MOVE_DIR_X.value < len(sensory_values):
            # Map -1, 0, 1 to 0.0, 0.5, 1.0
            sensory_values[Sensor.LAST_MOVE_DIR_X.value] = float(last_dx) * 0.5 + 0.5

        if Sensor.LAST_MOVE_DIR_Y.value < len(sensory_values):
            # Map -1, 0, 1 to 0.0, 0.5, 1.0
            sensory_values[Sensor.LAST_MOVE_DIR_Y.value] = float(last_dy) * 0.5 + 0.5

        # Long probe sensors (use discrete offset for direction)
        if Sensor.LONGPROBE_POP_FWD.value < len(sensory_values):
            probe_dist = self.get_long_probe_population(grid, last_dx, last_dy)
            sensory_values[Sensor.LONGPROBE_POP_FWD.value] = probe_dist / self.longprobe_dist

        if Sensor.LONGPROBE_BAR_FWD.value < len(sensory_values):
            probe_dist = self.get_long_probe_barrier(grid, last_dx, last_dy)
            sensory_values[Sensor.LONGPROBE_BAR_FWD.value] = probe_dist / self.longprobe_dist

        # Population density
        if Sensor.POPULATION.value < len(sensory_values):
            density = self.get_population_density(grid)
            sensory_values[Sensor.POPULATION.value] = density

        # Population density axis (use discrete offset for direction)
        if Sensor.POPULATION_FWD.value < len(sensory_values):
            density = self.get_population_density_axis(grid, (last_dx, last_dy))
            sensory_values[Sensor.POPULATION_FWD.value] = density

        if Sensor.POPULATION_LR.value < len(sensory_values):
            # Perpendicular to last move direction
            perp_x, perp_y = -last_dy, last_dx
            density = self.get_population_density_axis(grid, (perp_x, perp_y))
            sensory_values[Sensor.POPULATION_LR.value] = density

        # Barrier detection (use discrete offset for direction)
        if Sensor.BARRIER_FWD.value < len(sensory_values):
            barrier_dist = self.get_short_probe_barrier(grid, (last_dx, last_dy))
            sensory_values[Sensor.BARRIER_FWD.value] = barrier_dist

        if Sensor.BARRIER_LR.value < len(sensory_values):
            # Perpendicular to last move direction
            perp_x, perp_y = -last_dy, last_dx
            barrier_dist = self.get_short_probe_barrier(grid, (perp_x, perp_y))
            sensory_values[Sensor.BARRIER_LR.value] = barrier_dist

        # Age
        if Sensor.AGE.value < len(sensory_values):
            sensory_values[Sensor.AGE.value] = self.age / self.params['max_age']

        # Oscillator (match C++: use simStep and -cos)
        if Sensor.OSC1.value < len(sensory_values):
            phase = (sim_step % self.oscPeriod) / float(self.oscPeriod) # Use sim_step
            factor = -math.cos(phase * 2.0 * math.pi) # Use -cos
            sensor_val = (factor + 1.0) / 2.0 # Convert to 0.0..1.0
            # Clip any round-off error
            sensory_values[Sensor.OSC1.value] = min(1.0, max(0.0, sensor_val))

        # Random
        if Sensor.RANDOM.value < len(sensory_values):
            sensory_values[Sensor.RANDOM.value] = random.random()

        # Signal (pheromone) sensors
        if Sensor.SIGNAL0.value < len(sensory_values) and signals is not None:
            sensory_values[Sensor.SIGNAL0.value] = grid.get_pheromone_level(int(x), int(y))

        # Signal (pheromone) sensors (use discrete offset for direction)
        if Sensor.SIGNAL0.value < len(sensory_values) and signals is not None:
            # Use get_signal_density helper matching C++
            sensory_values[Sensor.SIGNAL0.value] = self.get_signal_density(signals, 0, grid)

        if Sensor.SIGNAL0_FWD.value < len(sensory_values) and signals is not None:
            # Use get_signal_density_along_axis helper matching C++
            sensory_values[Sensor.SIGNAL0_FWD.value] = self.get_signal_density_along_axis(signals, 0, grid, (last_dx, last_dy))

        if Sensor.SIGNAL0_LR.value < len(sensory_values) and signals is not None:
            # Use get_signal_density_along_axis helper matching C++
            perp_x, perp_y = -last_dy, last_dx
            sensory_values[Sensor.SIGNAL0_LR.value] = self.get_signal_density_along_axis(signals, 0, grid, (perp_x, perp_y))

        # Genetic similarity with creature in front (use discrete offset for direction)
        if Sensor.GENETIC_SIM_FWD.value < len(sensory_values):
            front_x = min(self.params['world_size'][0] - 1, max(0, int(x + last_dx)))
            front_y = min(self.params['world_size'][1] - 1, max(0, int(y + last_dy)))
            target_id = int(grid.data[front_x, front_y, 0])

            if target_id > 0 and target_id != self.id:
                for other in creatures:
                    if other.id == target_id:
                        similarity = calculate_genetic_similarity(self.genome, other.genome)
                        sensory_values[Sensor.GENETIC_SIM_FWD.value] = similarity
                        break
            else:
                sensory_values[Sensor.GENETIC_SIM_FWD.value] = 0.0

        # Removed Python-specific sensors: SAFE_ZONE, HAZARD_ZONE, RADIATION
        # Ensure Sensor enum and num_sensory_neurons match C++

        return sensory_values

    def get_long_probe_population(self, grid, dx, dy):
        """Get distance to nearest creature in the forward direction"""
        max_dist = self.longprobe_dist
        x, y = self.position

        for d in range(1, max_dist + 1):
            probe_x = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            probe_y = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            if probe_x == int(x) and probe_y == int(y):
                # No movement in this direction
                break

            if grid.data[probe_x, probe_y, 0] > 0:  # Found a creature
                return d

            if d == max_dist:  # Reached max distance without finding creature
                return max_dist

        return max_dist

    def get_long_probe_barrier(self, grid, dx, dy):
        """Get distance to nearest barrier in the forward direction"""
        max_dist = self.longprobe_dist
        x, y = self.position

        for d in range(1, max_dist + 1):
            probe_x = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            probe_y = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            if probe_x == int(x) and probe_y == int(y):
                # No movement in this direction
                break

            if grid.is_barrier_at(probe_x, probe_y):  # Found a barrier
                return d

            if d == max_dist:  # Reached max distance without finding barrier
                return max_dist

        return max_dist

    def get_population_density(self, grid):
        """Get local population density in neighborhood"""
        radius = self.params.get('populationSensorRadius', 2.5)
        x, y = self.position

        count = 0
        total = 0

        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                dist = dx * dx + dy * dy
                if dist <= radius * radius:  # Within circular radius
                    nx = min(self.params['world_size'][0] - 1, max(0, int(x + dx)))
                    ny = min(self.params['world_size'][1] - 1, max(0, int(y + dy)))

                    if grid.data[nx, ny, 0] > 0:  # Contains a creature
                        count += 1
                    total += 1

        return count / total if total > 0 else 0

    def get_population_density_axis(self, grid, direction):
        """Get population density gradient along an axis, matching C++"""
        dx, dy = direction
        # Handle zero direction vector (e.g., first step)
        if dx == 0 and dy == 0:
            return 0.5 # Return mid-range if no direction

        radius = self.params.get('population_sensor_radius', 2.5) # Use Python param name
        x, y = self.position

        # Normalize the direction vector
        dir_len = math.sqrt(dx * dx + dy * dy)
        if dir_len == 0: return 0.5 # Should not happen with check above, but safety
        dir_vec_x = dx / dir_len
        dir_vec_y = dy / dir_len

        sum_val = 0.0

        # Use grid's visit_neighborhood for efficiency and consistency
        def visit_callback(loc):
            nonlocal sum_val
            tloc_x, tloc_y = loc
            if (tloc_x, tloc_y) != (int(x), int(y)) and grid.isOccupiedAt(tloc_x, tloc_y):
                offset_x = tloc_x - x
                offset_y = tloc_y - y
                dist_sq = offset_x * offset_x + offset_y * offset_y
                if dist_sq > 0: # Avoid division by zero
                    # Projection magnitude along the direction axis
                    proj = dir_vec_x * offset_x + dir_vec_y * offset_y
                    # C++ calculation: proj / dist_sq
                    contrib = proj / dist_sq
                    sum_val += contrib

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        # C++ normalization
        max_sum_mag = 6.0 * radius
        if max_sum_mag == 0: return 0.5 # Avoid division by zero

        sensor_val = sum_val / max_sum_mag # convert to approx -1.0..1.0
        sensor_val = (sensor_val + 1.0) / 2.0 # convert to 0.0..1.0

        # Clip to ensure valid range
        return max(0.0, min(1.0, sensor_val))

    def get_short_probe_barrier(self, grid, direction):
        """Get barrier distance along an axis (forward and reverse)"""
        dx, dy = direction
        probe_distance = self.params.get('shortProbeBarrierDistance', 4)
        x, y = self.position

        # Search forward
        forward_dist = 0
        for d in range(1, probe_distance + 1):
            nx = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            ny = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            if grid.is_barrier_at(nx, ny):
                break
            forward_dist += 1

            if nx == 0 or nx == self.params['world_size'][0] - 1 or ny == 0 or ny == self.params['world_size'][1] - 1:
                forward_dist = probe_distance
                break

        # Search backward
        backward_dist = 0
        for d in range(1, probe_distance + 1):
            nx = min(self.params['world_size'][0] - 1, max(0, int(x - dx * d)))
            ny = min(self.params['world_size'][1] - 1, max(0, int(y - dy * d)))

            if grid.is_barrier_at(nx, ny):
                break
            backward_dist += 1

            if nx == 0 or nx == self.params['world_size'][0] - 1 or ny == 0 or ny == self.params['world_size'][1] - 1:
                backward_dist = probe_distance
                break

        # Normalize to range 0..1 based on forward-backward gradient
        return ((forward_dist - backward_dist) + probe_distance) / (2.0 * probe_distance)

    def get_signal_density(self, signals, layer_num, grid):
        """Get signal density in the neighborhood, matching C++"""
        radius = self.params.get('signal_sensor_radius', 2.5) # Use Python param name
        x, y = self.position
        
        count_locs = 0
        signal_sum = 0.0
        
        # Use grid's visit_neighborhood
        def visit_callback(loc):
            nonlocal count_locs, signal_sum
            tloc_x, tloc_y = loc
            count_locs += 1
            signal_sum += signals.get_value(layer_num, tloc_x, tloc_y) # Assumes signals.get_value exists

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)
        
        if count_locs == 0: return 0.0
        
        # C++ normalization: sum / (countLocs * SIGNAL_MAX)
        # Assuming SIGNAL_MAX is 1.0 for Python float signals
        max_signal_sum = float(count_locs) * 1.0
        if max_signal_sum == 0: return 0.0
        
        sensor_val = signal_sum / max_signal_sum # convert to 0.0..1.0
        return max(0.0, min(1.0, sensor_val)) # Clip

    def get_signal_density_along_axis(self, signals, layer_num, grid, direction):
        """Get signal density gradient along an axis, matching C++"""
        dx, dy = direction
        # Handle zero direction vector
        if dx == 0 and dy == 0:
            return 0.5

        radius = self.params.get('signal_sensor_radius', 2.5) # Use Python param name
        x, y = self.position

        # Normalize the direction vector
        dir_len = math.sqrt(dx * dx + dy * dy)
        if dir_len == 0: return 0.5
        dir_vec_x = dx / dir_len
        dir_vec_y = dy / dir_len

        sum_val = 0.0

        # Use grid's visit_neighborhood
        def visit_callback(loc):
            nonlocal sum_val
            tloc_x, tloc_y = loc
            if (tloc_x, tloc_y) != (int(x), int(y)):
                offset_x = tloc_x - x
                offset_y = tloc_y - y
                dist_sq = offset_x * offset_x + offset_y * offset_y
                if dist_sq > 0:
                    # Projection magnitude along the direction axis
                    proj = dir_vec_x * offset_x + dir_vec_y * offset_y
                    # C++ calculation: (proj * signal_magnitude) / dist_sq
                    signal_magnitude = signals.get_value(layer_num, tloc_x, tloc_y)
                    contrib = (proj * signal_magnitude) / dist_sq
                    sum_val += contrib

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        # C++ normalization
        # Assuming SIGNAL_MAX is 1.0 for Python float signals
        max_sum_mag = 6.0 * radius * 1.0
        if max_sum_mag == 0: return 0.5

        sensor_val = sum_val / max_sum_mag # convert to approx -1.0..1.0
        sensor_val = (sensor_val + 1.0) / 2.0 # convert to 0.0..1.0

        # Clip to ensure valid range
        return max(0.0, min(1.0, sensor_val))

    def process_actions(self, action_values, grid, creatures=None, signals=None):
        """
        Process neural network action values, implementing comprehensive action mechanics

        Args:
            action_values: Neural network output values
            grid: World grid
            creatures: List of all creatures
            signals: Signal/pheromone manager
        """
        # Ensure action_values is not None and has correct length
        if action_values is None:
            action_values = np.zeros(self.params['num_output_neurons'])

        # Responsiveness calculation (affects all actions)
        responsiveness = 1.0
        if Action.SET_RESPONSIVENESS.value < len(action_values):
            responsiveness = action_values[Action.SET_RESPONSIVENESS.value]
            responsiveness = max(0.0, min(1.0, responsiveness))
        self.responsiveness = responsiveness

        # First ensure action_values is a numpy array, not a dictionary
        if isinstance(action_values, dict):
            # Convert dictionary to numpy array if needed
            values_array = np.zeros(self.params['num_output_neurons'])
            for action, value in action_values.items():
                if isinstance(action, int) and action < len(values_array):
                    values_array[action] = value
            action_values = values_array

        # Movement Accumulation (similar to previous implementation)
        move_x = action_values[Action.MOVE_X.value] * 2 - 1 if Action.MOVE_X.value < len(action_values) else 0
        move_y = action_values[Action.MOVE_Y.value] * 2 - 1 if Action.MOVE_Y.value < len(action_values) else 0

        # Add cardinal and directional movement components
        if Action.MOVE_EAST.value < len(action_values):
            move_x += action_values[Action.MOVE_EAST.value]
        if Action.MOVE_WEST.value < len(action_values):
            move_x -= action_values[Action.MOVE_WEST.value]
        if Action.MOVE_NORTH.value < len(action_values):
            move_y += action_values[Action.MOVE_NORTH.value]
        if Action.MOVE_SOUTH.value < len(action_values):
            move_y -= action_values[Action.MOVE_SOUTH.value]

        # Relative movement based on current direction
        if Action.MOVE_FORWARD.value < len(action_values):
            forward_level = action_values[Action.MOVE_FORWARD.value]
            move_x += self.direction[0] * forward_level
            move_y += self.direction[1] * forward_level

        if Action.MOVE_REVERSE.value < len(action_values):
            reverse_level = action_values[Action.MOVE_REVERSE.value]
            move_x -= self.direction[0] * reverse_level
            move_y -= self.direction[1] * reverse_level

        # Sideways movement
        if Action.MOVE_LEFT.value < len(action_values):
            left_level = action_values[Action.MOVE_LEFT.value]
            move_x -= self.direction[1] * left_level
            move_y += self.direction[0] * left_level

        if Action.MOVE_RIGHT.value < len(action_values):
            right_level = action_values[Action.MOVE_RIGHT.value]
            move_x += self.direction[1] * right_level
            move_y -= self.direction[0] * right_level

        # Random movement option
        if Action.MOVE_RANDOM.value < len(action_values):
            random_level = action_values[Action.MOVE_RANDOM.value]
            random_angle = random.uniform(0, 2 * math.pi)
            move_x += math.cos(random_angle) * random_level
            move_y += math.sin(random_angle) * random_level

        # Scale movement by responsiveness
        move_x *= responsiveness
        move_y *= responsiveness

        # Discrete grid movement with probabilistic selection
        dx, dy = 0, 0
        if random.random() < abs(move_x):
            dx = 1 if move_x > 0 else -1
        if random.random() < abs(move_y):
            dy = 1 if move_y > 0 else -1

        # Calculate new position with boundary checking
        new_x = max(0, min(self.params['world_size'][0] - 1, self.position[0] + dx))
        new_y = max(0, min(self.params['world_size'][1] - 1, self.position[1] + dy))

        # Grid-based movement
        if (new_x, new_y) != self.position and grid.is_empty_at(int(new_x), int(new_y)):
            # Clear old position
            grid.data[int(self.position[0]), int(self.position[1]), 0] = 0

            # Update direction based on movement
            if dx != 0 or dy != 0:
                length = math.sqrt(dx * dx + dy * dy)
                self.direction = (dx / length, dy / length)

            # Move creature
            self.position = (new_x, new_y)
            grid.data[int(new_x), int(new_y), 0] = self.id

        # Oscillator period setting
        if Action.SET_OSCILLATOR_PERIOD.value < len(action_values):
            periodf = action_values[Action.SET_OSCILLATOR_PERIOD.value]
            newPeriodf01 = (math.tanh(periodf) + 1.0) / 2.0
            newPeriod = 1 + int(1.5 + math.exp(7.0 * newPeriodf01))
            self.oscPeriod = max(2, min(newPeriod, 2048))

        # Long probe distance setting
        if Action.SET_LONGPROBE_DIST.value < len(action_values):
            max_probe_dist = 32
            level = action_values[Action.SET_LONGPROBE_DIST.value]
            level = (math.tanh(level) + 1.0) / 2.0
            probe_dist = 1 + int(level * max_probe_dist)
            self.longprobe_dist = probe_dist

        # Signal emission
        if Action.EMIT_SIGNAL0.value < len(action_values):
            emit_threshold = 0.5
            level = action_values[Action.EMIT_SIGNAL0.value]
            level = (math.tanh(level) + 1.0) / 2.0
            level *= responsiveness

            if level > emit_threshold and random.random() < level:
                signals.increment(0, int(self.position[0]), int(self.position[1]))

        # Kill forward action (if enabled)
        if (self.params.get('enable_kill_neuron', False) and
                Action.KILL_FORWARD.value < len(action_values)):
            kill_threshold = 0.5
            level = action_values[Action.KILL_FORWARD.value]
            level = (math.tanh(level) + 1.0) / 2.0
            level *= responsiveness

            if level > kill_threshold and random.random() < level:
                # Calculate position in front of creature
                front_x = int(min(self.params['world_size'][0] - 1,
                                  max(0, self.position[0] + self.direction[0])))
                front_y = int(min(self.params['world_size'][1] - 1,
                                  max(0, self.position[1] + self.direction[1])))

                # Check if there's a creature at that position
                target_id = int(grid.data[front_x, front_y, 0])
                if target_id > 0 and target_id != self.id:
                    grid.queue_for_death(target_id)
                    self.has_killed = True

                    # Optional: Energy gain from kill
                    for other in creatures:
                        if other.id == target_id:
                            energy_gain = min(other.energy * 0.5, 200)
                            self.energy += energy_gain
                            break

        # Apply metabolic costs
        self.energy -= self.metabolic_rate * self.energy_efficiency
        self.energy = max(0.0, self.energy)

    def handle_boundary_effects(self, dx, dy):
        """
        Apply small inward force when near boundaries to prevent edge-hugging
        """
        # Distance from each boundary
        dist_left = self.position[0]
        dist_right = self.params['world_size'][0] - 1 - self.position[0]
        dist_bottom = self.position[1]
        dist_top = self.params['world_size'][1] - 1 - self.position[1]

        # Boundary avoidance range (percentage of world size)
        boundary_range = min(self.params['world_size']) * 0.05

        # Apply gradual inward force based on proximity to edges
        if dist_left < boundary_range:
            # Near left edge, push right
            nudge = (boundary_range - dist_left) / boundary_range
            dx += nudge * 0.3
        elif dist_right < boundary_range:
            # Near right edge, push left
            nudge = (boundary_range - dist_right) / boundary_range
            dx -= nudge * 0.3

        if dist_bottom < boundary_range:
            # Near bottom edge, push up
            nudge = (boundary_range - dist_bottom) / boundary_range
            dy += nudge * 0.3
        elif dist_top < boundary_range:
            # Near top edge, push down
            nudge = (boundary_range - dist_top) / boundary_range
            dy -= nudge * 0.3

        return dx, dy

    def update_health(self):
        """Update the creature's health based on various factors"""
        # Health decreases with age
        age_factor = 1.0 - (self.age / self.params['max_age']) * 0.5

        # Health affected by energy levels
        energy_factor = min(1.0, self.energy / 500)

        # Calculate new health value (weighted average)
        self.health = 0.7 * age_factor + 0.3 * energy_factor
        self.health = max(0.0, min(1.0, self.health))

        # Check if creature should die from old age or starvation
        if self.age >= self.params['max_age'] or self.energy <= 0:
            self.alive = False

    def detect_environment(self, grid, creatures):
        """Detect and record environmental information"""
        x, y = int(self.position[0]), int(self.position[1])

        # Record current zone type
        self.in_safe_zone = (grid.data[x, y, 2] == 1)

        # Record nearby creatures
        self.detected_creatures = []
        search_radius = 3
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < grid.size[0] and 0 <= ny < grid.size[1]:
                    creature_id = int(grid.data[nx, ny, 0])
                    if creature_id > 0 and creature_id != self.id:
                        self.detected_creatures.append((creature_id, (nx, ny)))

    def move(self, grid, creatures, signals=None):
        """
        Update the creature's position based on neural network action_values
        
        This method doesn't directly modify the grid, but instead queues
        movement requests that will be processed at the end of the step.
        """
        # Get sensor inputs using the comprehensive method
        sensor_values = self.get_sensory_inputs(grid, creatures, signals)

        # Create state dictionary
        creature_state = {
            'direction': self.direction,
            'responsiveness': self.responsiveness,
            'oscPeriod': self.oscPeriod,
            'longprobe_dist': self.longprobe_dist
        }
        
        # Process neural network
        action_values, state_updates = self.brain.feed_forward(
            sensor_values, self.age, creature_state)
        
        # Apply state updates
        self._apply_state_updates(state_updates, signals, grid, creatures)
        
        # Process and apply movement
        movement_info = self.brain.process_movement(action_values, creature_state)
        self.apply_movement(movement_info, grid)
        
        # Update age
        self.age += 1

    def _attempt_kill(self, grid, creatures):
        """Handle kill attempt action from neural network"""
        # Calculate position in front of creature
        front_x = int(min(self.params['world_size'][0] - 1,
                          max(0, self.position[0] + self.direction[0])))
        front_y = int(min(self.params['world_size'][1] - 1,
                          max(0, self.position[1] + self.direction[1])))

        # Check if there's a creature at that position
        target_id = int(grid.data[front_x, front_y, 0])
        if target_id > 0 and target_id != self.id:
            # Queue target for death
            grid.queue_for_death(target_id)
            self.has_killed = True

            # Find target creature to calculate energy gain
            for other in creatures:
                if other.id == target_id:
                    # Energy gain from kill - cap to prevent excessive energy
                    energy_gain = min(other.energy * 0.5, 200)
                    self.energy += energy_gain
                    break

    # _apply_movement method removed - consolidated into apply_movement

    # No need for redundant logging configuration - using centralized logging utility
