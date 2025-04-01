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

        self.genome = genome if genome else Genome(length=params['genome_length'], params=params) # Pass params
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
        # Removing age as it's not needed and not in biosim4
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
        self.longprobe_dist = params.get('longprobe_dist', 16) # Use python param name
        self.recent_signals = []  # Track recently detected signals
        self.detected_creatures = []  # Track recently detected creatures

        # Neural network characteristics
        self.responsiveness = 0.5  # Affects how strongly actions are executed (0.0-1.0)
        self.oscPeriod = 34  # Period for oscillator sensor

        # Initialize the neural network (brain)
        try:
            self.brain = NeuralNetwork(self.genome, params)
            
            # Ensure the creature has at least one valid connection
            if not self.brain.connections:
                # Add a direct connection from a random sensor to a movement action
                self.brain.connections.append({
                    'source_type': 0,  # SENSOR
                    'source_num': random.randint(0, params['num_sensory_neurons'] - 1),
                    'sink_type': 1,  # ACTION
                    'sink_num': random.randint(0, min(4, params['num_output_neurons'] - 1)),  # Movement action
                    'weight': 0.5  # Moderate weight
                })
                self.brain.total_connections = len(self.brain.connections)
        except Exception as e:
            logger.error(f"Error initializing neural network for creature {self.id}: {e}")
            # Initialize with a simple default neural network
            self.brain = NeuralNetwork(Genome(length=1, params=params), params)
            # Add a direct connection from a random sensor to a movement action
            self.brain.connections.append({
                'source_type': 0,  # SENSOR
                'source_num': random.randint(0, params['num_sensory_neurons'] - 1),
                'sink_type': 1,  # ACTION
                'sink_num': random.randint(0, min(4, params['num_output_neurons'] - 1)),  # Movement action
                'weight': 0.5  # Moderate weight
            })
            self.brain.total_connections = len(self.brain.connections)

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
        
        # Debug output for movement
        if dx == 0 and dy == 0:
            return False  # No movement

        # Apply boundary checks
        new_x = max(0, min(self.params['world_size'][0] - 1, int(self.position[0] + dx))) # Ensure int
        new_y = max(0, min(self.params['world_size'][1] - 1, int(self.position[1] + dy))) # Ensure int

        # Check if the target position is a barrier
        if grid.is_barrier_at(new_x, new_y): # Use int coords
            return False # Cannot move into a barrier
            
        # Check if the target position is already occupied by another creature
        if grid.data[new_x, new_y, 0] > 0 and grid.data[new_x, new_y, 0] != self.id:
            return False # Cannot move into an occupied cell

        # Update direction if provided
        if movement_info['direction_update']:
            self.direction = movement_info['direction_update']

        # Queue the movement for processing at the end of the step
        # This prevents race conditions where multiple creatures try to move to the same cell
        grid.queue_for_move(self.id, (new_x, new_y))
        
        # DO NOT update the creature's position directly
        # This will be done by the grid.process_move_queue method
        # self.position = (new_x, new_y)
        # self.last_move_offset = (dx, dy)

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
            elif key in ['oscPeriod', 'longprobe_dist', 'responsiveness']: # Check if key is a known attribute
                 setattr(self, key, value)

    def update(self, grid, creatures, signals, sim_step): # Add sim_step parameter
        """Update the creature for one simulation step"""
        if not self.alive:
            return

        # Check for zone effects at current position
        x, y = int(self.position[0]), int(self.position[1])
        zone_type = grid.data[x, y, 2]  # Get zone type from grid

        # Update creature's zone status and apply zone effects
        self._process_zone_effects(zone_type)

        # Get sensory inputs based on current environment, passing sim_step
        sensory_inputs = self.get_sensory_inputs(grid, creatures, signals, sim_step) # Use passed sim_step

        # Create state dictionary to pass creature context to neural network
        creature_state = {
            'direction': self.direction,
            'responsiveness': self.responsiveness,
            'oscPeriod': self.oscPeriod,
            'longprobe_dist': self.longprobe_dist
        }

        # Process neural network with enhanced state interaction
        # Pass sim_step to feed_forward
        action_values, state_updates = self.brain.feed_forward(
            sensory_inputs, sim_step, creature_state)

        # Apply neural network state updates
        self._apply_state_updates(state_updates, signals, grid, creatures)

        # Process movement using the consolidated logic
        movement_info = self.brain.process_movement(action_values, creature_state)
        self.apply_movement(movement_info, grid)

        # Update health and detect environment
        self.update_health()
        self.detect_environment(grid, creatures)

    def update_health(self):
        """Update the creature's health based on energy levels"""
        # Health affected by energy levels
        energy_factor = min(1.0, self.energy / 500)

        # Calculate new health value
        self.health = energy_factor
        self.health = max(0.0, min(1.0, self.health))

        # Check if creature should die from starvation
        if self.energy <= 0:
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
            max_x = self.params['world_size'][0] / 2.0 # Ensure float division
            sensory_values[Sensor.BOUNDARY_DIST_X.value] = dist_x_min / max_x if max_x > 0 else 0.0

        if Sensor.BOUNDARY_DIST_Y.value < len(sensory_values):
            dist_y_min = min(y, self.params['world_size'][1] - 1 - y)
            max_y = self.params['world_size'][1] / 2.0 # Ensure float division
            sensory_values[Sensor.BOUNDARY_DIST_Y.value] = dist_y_min / max_y if max_y > 0 else 0.0

        if Sensor.BOUNDARY_DIST.value < len(sensory_values):
            dist_x_min = min(x, self.params['world_size'][0] - 1 - x)
            dist_y_min = min(y, self.params['world_size'][1] - 1 - y)
            closest = min(dist_x_min, dist_y_min)
            # Ensure max_possible is at least 1 to avoid division by zero
            max_possible = max(1.0, self.params['world_size'][0] / 2.0 - 1,
                               self.params['world_size'][1] / 2.0 - 1)
            sensory_values[Sensor.BOUNDARY_DIST.value] = closest / max_possible

        # Last movement direction (use discrete offset)
        if Sensor.LAST_MOVE_DIR_X.value < len(sensory_values):
            # Map -1, 0, 1 to 0.0, 0.5, 1.0
            sensory_values[Sensor.LAST_MOVE_DIR_X.value] = float(last_dx) * 0.5 + 0.5

        if Sensor.LAST_MOVE_DIR_Y.value < len(sensory_values):
            # Map -1, 0, 1 to 0.0, 0.5, 1.0
            sensory_values[Sensor.LAST_MOVE_DIR_Y.value] = float(last_dy) * 0.5 + 0.5

        # Oscillator (match C++: use simStep and -cos)
        if Sensor.OSC1.value < len(sensory_values):
            # Ensure oscPeriod is not zero
            osc_period = float(self.oscPeriod) if self.oscPeriod > 0 else 1.0
            phase = (sim_step % self.oscPeriod) / osc_period # Use sim_step
            factor = -math.cos(phase * 2.0 * math.pi) # Use -cos
            sensor_val = (factor + 1.0) / 2.0 # Convert to 0.0..1.0
            # Clip any round-off error
            sensory_values[Sensor.OSC1.value] = min(1.0, max(0.0, sensor_val))

        # Random
        if Sensor.RANDOM.value < len(sensory_values):
            sensory_values[Sensor.RANDOM.value] = random.random()

        # Final check for NaN or out-of-range values
        if np.any(np.isnan(sensory_values)) or np.any(sensory_values < -0.01) or np.any(sensory_values > 1.01):
            logger.warning(f"Sensor values out of range or NaN for creature {self.id}: {sensory_values}")
            sensory_values = np.clip(sensory_values, 0.0, 1.0) # Clip to valid range [0, 1]

        return sensory_values

    def get_long_probe_population(self, grid, dx, dy):
        """Get distance to nearest creature in the forward direction"""
        # Handle zero direction vector
        if dx == 0 and dy == 0:
            return float(self.longprobe_dist)

        max_dist = self.longprobe_dist
        x, y = self.position

        for d in range(1, max_dist + 1):
            probe_x = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            probe_y = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            # Stop if probe hits barrier or another creature
            if grid.is_barrier_at(probe_x, probe_y):
                 return float(max_dist) # C++ returns max distance if barrier hit
            if grid.data[probe_x, probe_y, 0] > 0:  # Found a creature
                return float(d)

            # Stop if probe hits world boundary (implicit barrier)
            if probe_x == 0 or probe_x == self.params['world_size'][0] - 1 or \
               probe_y == 0 or probe_y == self.params['world_size'][1] - 1:
                 if d < max_dist: # Only return max_dist if boundary hit before max probe distance
                     return float(max_dist)

        return float(max_dist) # Reached max distance

    def get_long_probe_barrier(self, grid, dx, dy):
        """Get distance to nearest barrier in the forward direction"""
         # Handle zero direction vector
        if dx == 0 and dy == 0:
            return float(self.longprobe_dist)

        max_dist = self.longprobe_dist
        x, y = self.position

        for d in range(1, max_dist + 1):
            probe_x = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            probe_y = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            if grid.is_barrier_at(probe_x, probe_y):  # Found a barrier
                return float(d)

            # Stop if probe hits world boundary
            if probe_x == 0 or probe_x == self.params['world_size'][0] - 1 or \
               probe_y == 0 or probe_y == self.params['world_size'][1] - 1:
                 if d < max_dist: # C++ returns max distance if boundary hit before max probe distance
                     return float(max_dist)

        return float(max_dist) # Reached max distance

    def get_population_density(self, grid):
        """Get local population density in neighborhood"""
        radius = self.params.get('population_sensor_radius', 2.5) # Use Python param name
        x, y = self.position

        count_occupied = 0
        count_locs = 0

        # Use grid's visit_neighborhood
        def visit_callback(loc):
            nonlocal count_occupied, count_locs
            tloc_x, tloc_y = loc
            count_locs += 1
            if grid.isOccupiedAt(tloc_x, tloc_y):
                 count_occupied += 1

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        return float(count_occupied) / count_locs if count_locs > 0 else 0.0

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
            # C++ includes the center location in the neighborhood check but excludes it from the sum calculation
            if grid.isOccupiedAt(tloc_x, tloc_y) and (tloc_x, tloc_y) != (int(x), int(y)):
                offset_x = float(tloc_x - x)
                offset_y = float(tloc_y - y)
                dist_sq = offset_x * offset_x + offset_y * offset_y
                if dist_sq > 1e-6: # Avoid division by zero or near-zero
                    # Projection magnitude along the direction axis
                    proj = dir_vec_x * offset_x + dir_vec_y * offset_y
                    # C++ calculation: proj / dist_sq
                    contrib = proj / dist_sq
                    sum_val += contrib

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        # C++ normalization
        max_sum_mag = 6.0 * radius
        if max_sum_mag < 1e-6: return 0.5 # Avoid division by zero

        sensor_val = sum_val / max_sum_mag # convert to approx -1.0..1.0
        sensor_val = (sensor_val + 1.0) / 2.0 # convert to 0.0..1.0

        # Clip to ensure valid range
        return max(0.0, min(1.0, sensor_val))

    def get_short_probe_barrier(self, grid, direction):
        """Get barrier distance along an axis (forward and reverse)"""
        dx, dy = direction
        # Handle zero direction vector
        if dx == 0 and dy == 0:
            return 0.5 # Return mid-range if no direction

        probe_distance = self.params.get('short_probe_barrier_distance', 4) # Use Python param name
        x, y = self.position

        # Search forward
        forward_dist = 0
        for d in range(1, probe_distance + 1):
            nx = min(self.params['world_size'][0] - 1, max(0, int(x + dx * d)))
            ny = min(self.params['world_size'][1] - 1, max(0, int(y + dy * d)))

            if grid.is_barrier_at(nx, ny):
                break
            forward_dist += 1
            # C++ logic: if boundary hit before probe distance, count as max distance
            if nx == 0 or nx == self.params['world_size'][0] - 1 or ny == 0 or ny == self.params['world_size'][1] - 1:
                 if d < probe_distance:
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
            # C++ logic: if boundary hit before probe distance, count as max distance
            if nx == 0 or nx == self.params['world_size'][0] - 1 or ny == 0 or ny == self.params['world_size'][1] - 1:
                 if d < probe_distance:
                     backward_dist = probe_distance
                     break

        # Normalize to range 0..1 based on forward-backward gradient
        denominator = 2.0 * probe_distance
        if denominator == 0: return 0.5 # Avoid division by zero
        sensor_val = ((forward_dist - backward_dist) + probe_distance) / denominator
        return max(0.0, min(1.0, sensor_val)) # Clip

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
            # C++ uses uint8_t (0-255), Python uses float (0.0-1.0)
            # We assume signals.get_value returns 0.0-1.0
            signal_sum += signals.get_value(layer_num, tloc_x, tloc_y)

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        if count_locs == 0: return 0.0

        # C++ normalization: sum / (countLocs * SIGNAL_MAX)
        # Python uses float 0.0-1.0, so SIGNAL_MAX equivalent is 1.0
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
        if dir_len < 1e-6: return 0.5 # Avoid division by zero if vector is tiny
        dir_vec_x = dx / dir_len
        dir_vec_y = dy / dir_len

        sum_val = 0.0

        # Use grid's visit_neighborhood
        def visit_callback(loc):
            nonlocal sum_val
            tloc_x, tloc_y = loc
            # C++ includes center, but excludes from sum calculation below
            if (tloc_x, tloc_y) != (int(x), int(y)):
                offset_x = float(tloc_x - x)
                offset_y = float(tloc_y - y)
                dist_sq = offset_x * offset_x + offset_y * offset_y
                if dist_sq > 1e-6: # Avoid division by zero
                    # Projection magnitude along the direction axis
                    proj = dir_vec_x * offset_x + dir_vec_y * offset_y
                    # C++ calculation: (proj * signal_magnitude) / dist_sq
                    # Python uses float 0.0-1.0 for signals
                    signal_magnitude = signals.get_value(layer_num, tloc_x, tloc_y)
                    contrib = (proj * signal_magnitude) / dist_sq
                    sum_val += contrib

        grid.visit_neighborhood((int(x), int(y)), radius, visit_callback)

        # C++ normalization
        # Python uses float 0.0-1.0, so SIGNAL_MAX equivalent is 1.0
        max_sum_mag = 6.0 * radius * 1.0
        if max_sum_mag < 1e-6: return 0.5 # Avoid division by zero

        sensor_val = sum_val / max_sum_mag # convert to approx -1.0..1.0
        sensor_val = (sensor_val + 1.0) / 2.0 # convert to 0.0..1.0

        # Clip to ensure valid range
        return max(0.0, min(1.0, sensor_val))
