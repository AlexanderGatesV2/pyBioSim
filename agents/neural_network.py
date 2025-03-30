import numpy as np
import math
import random
from core.types import Sensor, Action, calculate_genetic_similarity

class NeuralNetwork:
    def __init__(self, genome, params):
        """
        Initialize a neural network from a genome, matching C++ implementation

        Args:
            genome: The creature's genome defining neural connections
            params: Simulation parameters
        """
        # Add constants to match C++ implementation
        self.SENSOR_MIN = 0.0
        self.SENSOR_MAX = 1.0
        self.NEURON_MIN = -1.0
        self.NEURON_MAX = 1.0
        self.ACTION_MIN = 0.0
        self.ACTION_MAX = 1.0

        # Add support for action types lookup
        self.initialize_action_indices()

        self.genome = genome
        self.params = params

        # Initialize neurons with values from genome if available
        if hasattr(genome, 'initial_neuron_values'):
            # Use genome's neuron initialization values if available
            self.neurons = []
            for i in range(params['num_internal_neurons']):
                if i in genome.initial_neuron_values:
                    initial_value = genome.initial_neuron_values[i]
                else:
                    initial_value = 0.5  # Default value

                self.neurons.append({
                    'output': initial_value,
                    'driven': False  # Will be set during connection building
                })
        else:
            # Default initialization
            self.neurons = [{'output': 0.5, 'driven': False} for _ in range(params['num_internal_neurons'])]

        # Connection list will be built from genome
        self.connections = []

        # Build the connection list from genome
        self.build_connections_from_genome()

        # Track metrics
        self.active_internal_neurons = 0
        self.total_connections = len(self.connections)
        self.calculate_network_metrics()

    def build_connections_from_genome(self):
        """Build neural connections from genome, pruning useless neurons"""
        # Phase 1: Create initial connection list from genes
        connections_list = []
        for gene in self.genome.genes:
            # Normalize gene indices to valid ranges
            source_type = getattr(gene, 'source_type', gene['source_type'] if isinstance(gene, dict) else 0)
            source_num = getattr(gene, 'source_id', gene['source_id'] if isinstance(gene, dict) else 0)
            source_num %= (self.params['num_sensory_neurons'] if source_type == 0
                          else self.params['num_internal_neurons'])

            sink_type = getattr(gene, 'sink_type', gene['sink_type'] if isinstance(gene, dict) else 0)
            sink_num = getattr(gene, 'sink_id', gene['sink_id'] if isinstance(gene, dict) else 0)
            sink_num %= (self.params['num_internal_neurons'] if sink_type == 0
                        else self.params['num_output_neurons'])

            weight = getattr(gene, 'weight', gene['weight'] if isinstance(gene, dict) else 0.0)

            # Create connection with normalized indices
            connections_list.append({
                'source_type': source_type,
                'source_num': source_num,
                'sink_type': sink_type,
                'sink_num': sink_num,
                'weight': weight
            })

        # Phase 2: Find neurons with outputs and track which neurons are driven
        neuron_outputs = [0] * self.params['num_internal_neurons']
        neuron_has_external_input = [False] * self.params['num_internal_neurons']
        self_inputs_count = [0] * self.params['num_internal_neurons']

        # Count outputs and inputs in a single pass
        for conn in connections_list:
            if conn['sink_type'] == 0:  # NEURON
                # This neuron has an input
                if conn['source_type'] == 0:  # From SENSOR
                    neuron_has_external_input[conn['sink_num']] = True
                elif conn['source_type'] == 1:  # From NEURON
                    if conn['source_num'] == conn['sink_num']:
                        self_inputs_count[conn['sink_num']] += 1
                    else:
                        neuron_has_external_input[conn['sink_num']] = True

            if conn['source_type'] == 1:  # NEURON
                # This neuron has an output
                neuron_outputs[conn['source_num']] += 1

        # Phase 3: Identify useless neurons in a single pass
        useless_neurons = set()
        for i in range(self.params['num_internal_neurons']):
            # A neuron is useless if it has no outputs or only self-connections
            if neuron_outputs[i] == 0 or neuron_outputs[i] == self_inputs_count[i]:
                useless_neurons.add(i)

        # Phase 4: Remove connections to/from useless neurons
        # Continue until no more neurons become useless
        while useless_neurons:
            # Remove connections to/from these neurons
            new_connections = []
            new_useless = set()

            for conn in connections_list:
                # Skip connections to/from useless neurons
                if ((conn['source_type'] == 1 and conn['source_num'] in useless_neurons) or
                        (conn['sink_type'] == 0 and conn['sink_num'] in useless_neurons)):
                    # If this removes an output from a source neuron, that neuron might become useless
                    if (conn['source_type'] == 1 and conn['sink_type'] == 0 and
                            conn['source_num'] not in useless_neurons and
                            conn['sink_num'] not in useless_neurons):
                        neuron_outputs[conn['source_num']] -= 1
                        if neuron_outputs[conn['source_num']] == 0 or neuron_outputs[conn['source_num']] == \
                                self_inputs_count[conn['source_num']]:
                            new_useless.add(conn['source_num'])
                    continue
                new_connections.append(conn)

            # Update connections list and useless neurons
            connections_list = new_connections
            useless_neurons = new_useless
            if not new_useless:
                break

        # Create mapping from old to new indices
        used_neurons = set()
        for conn in connections_list:
            if conn['source_type'] == 1:  # NEURON
                used_neurons.add(conn['source_num'])
            if conn['sink_type'] == 0:  # NEURON
                used_neurons.add(conn['sink_num'])

        used_neurons_list = sorted(list(used_neurons))
        neuron_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_neurons_list)}

        # Phase 5: Create final connection list with renumbered neurons
        # First pass: connections to neurons (for optimal feed-forward)
        neuron_connections = []
        for conn in connections_list:
            if conn['sink_type'] == 0:  # NEURON
                new_conn = conn.copy()
                if new_conn['source_type'] == 1:  # NEURON
                    new_conn['source_num'] = neuron_mapping[new_conn['source_num']]
                new_conn['sink_num'] = neuron_mapping[new_conn['sink_num']]
                neuron_connections.append(new_conn)

        # Second pass: connections to actions
        action_connections = []
        for conn in connections_list:
            if conn['sink_type'] == 1:  # ACTION
                new_conn = conn.copy()
                if new_conn['source_type'] == 1:  # NEURON
                    new_conn['source_num'] = neuron_mapping[new_conn['source_num']]
                action_connections.append(new_conn)

        # Combine connections (neurons first, then actions)
        self.connections = neuron_connections + action_connections

        # Create neurons with proper driven flags
        self.neurons = []
        for i in range(len(used_neurons_list)):
            original_idx = used_neurons_list[i]
            self.neurons.append({
                'output': 0.5,  # Initial output value
                'driven': neuron_has_external_input[original_idx]
            })
        # print(f"DEBUG: Final neurons list (len={len(self.neurons)}): {self.neurons}") # DEBUG
        # print(f"DEBUG: Final connections list (len={len(self.connections)}): {self.connections}") # DEBUG


    def feed_forward(self, sensory_inputs, sim_step, creature_state=None):
        """
        Perform neural network feed-forward processing, matching C++ implementation.

        Args:
            sensory_inputs: Array of sensor values
            sim_step: Current simulation step
            creature_state: Optional dictionary with creature state info for advanced processing

        Returns:
            Tuple of (action_values, state_updates) where state_updates contains non-movement actions
        """
        # Initialize with zeros if no sensory inputs
        if sensory_inputs is None:
            sensory_inputs = np.zeros(self.params['num_sensory_neurons'])

        # Initialize action values to 0
        action_values = np.zeros(self.params['num_output_neurons'])

        # Early return for networks with no connections
        if not self.connections:
            return action_values, {}

        # Initialize neuron input accumulators (following the C++ implementation)
        neuron_inputs = np.zeros(len(self.neurons))

        # Track if we've computed neuron outputs yet
        neuron_outputs_computed = False

        # Process connections in the correct order (matching C++ ordering)
        # print(f"DEBUG: Starting feed_forward for genome {self.genome.hash()[:16]}...") # DEBUG
        # print(f"DEBUG: Initial neuron outputs: {[n['output'] for n in self.neurons]}") # DEBUG
        
        action_accumulators = np.zeros(self.params['num_output_neurons']) # Accumulate action inputs separately

        for conn_idx, conn in enumerate(self.connections):
            # print(f"DEBUG: Processing connection {conn_idx}: {conn}") # DEBUG
            # Process connections to neurons first
            if conn['sink_type'] == 0 and not neuron_outputs_computed:  # NEURON
                # Get input value from sensor or neuron
                if conn['source_type'] == 0:  # SENSOR
                    input_val = sensory_inputs[conn['source_num']]
                else:  # NEURON
                    input_val = self.neurons[conn['source_num']]['output']

                # Add weighted input to neuron accumulator
                # Use the pre-scaled float weight stored in the connection
                weight_float = conn['weight']
                delta = input_val * weight_float
                # print(f"DEBUG: Updating neuron_inputs[{conn['sink_num']}] += {delta} (input={input_val}, weight_float={weight_float})") # DEBUG
                neuron_inputs[conn['sink_num']] += delta
                # print(f"DEBUG: neuron_inputs[{conn['sink_num']}] is now {neuron_inputs[conn['sink_num']]}") # DEBUG

            # When we encounter the first action connection, compute all neuron outputs
            elif conn['sink_type'] == 1 and not neuron_outputs_computed:  # First ACTION connection
                # print("DEBUG: Computing neuron outputs...") # DEBUG
                # Update all neuron outputs with tanh activation (exactly matching C++)
                for i in range(len(self.neurons)):
                    if self.neurons[i]['driven']:
                        self.neurons[i]['output'] = math.tanh(neuron_inputs[i])
                        # print(f"DEBUG: Neuron {i} output = tanh({neuron_inputs[i]}) = {self.neurons[i]['output']}") # DEBUG
                    # Note: undriven neurons maintain their value from initialization

                neuron_outputs_computed = True
                # print(f"DEBUG: Computed neuron outputs: {[n['output'] for n in self.neurons]}") # DEBUG

                # Now process this connection (fall through to the action processing)
                if conn['source_type'] == 0:  # SENSOR
                    input_val = sensory_inputs[conn['source_num']]
                else:  # NEURON
                    input_val = self.neurons[conn['source_num']]['output']

                # Add weighted input to action value
                # Use the pre-scaled float weight stored in the connection
                weight_float = conn['weight']
                action_accumulators[conn['sink_num']] += input_val * weight_float
                # print(f"DEBUG: Action {conn['sink_num']} input += {input_val} * {weight_float} = {input_val * weight_float}. New total: {action_accumulators[conn['sink_num']]}") # DEBUG


            # Process remaining action connections
            elif conn['sink_type'] == 1:  # ACTION
                # Get input value from sensor or neuron
                if conn['source_type'] == 0:  # SENSOR
                    input_val = sensory_inputs[conn['source_num']]
                else:  # NEURON
                    input_val = self.neurons[conn['source_num']]['output']

                # Add weighted input to action value
                # Use the pre-scaled float weight stored in the connection
                weight_float = conn['weight']
                action_accumulators[conn['sink_num']] += input_val * weight_float
                # print(f"DEBUG: Action {conn['sink_num']} input += {input_val} * {weight_float} = {input_val * weight_float}. New total: {action_accumulators[conn['sink_num']]}") # DEBUG


        # If there were no action connections, compute neuron outputs now
        if not neuron_outputs_computed:
            # print("DEBUG: Computing neuron outputs (no action connections found)...") # DEBUG
            for i in range(len(self.neurons)):
                if self.neurons[i]['driven']:
                    self.neurons[i]['output'] = math.tanh(neuron_inputs[i])
                    # print(f"DEBUG: Neuron {i} output = tanh({neuron_inputs[i]}) = {self.neurons[i]['output']}") # DEBUG
            # print(f"DEBUG: Computed neuron outputs: {[n['output'] for n in self.neurons]}") # DEBUG


        # Apply tanh activation and scale to 0..1 range for action values
        # This matches the C++ implementation in executeActions.cpp
        # print(f"DEBUG: Action accumulators before tanh: {action_accumulators}") # DEBUG
        for i in range(len(action_values)):
            action_values[i] = (math.tanh(action_accumulators[i]) + 1.0) / 2.0 # Use accumulators

        # Process specialized movement actions (matching C++ executeActions.cpp)
        state_updates = self._post_process_actions(action_values, creature_state)

        return action_values, state_updates

    def calculate_network_metrics(self):
        """Calculate metrics about the neural network"""
        # Count active neurons (those with connections)
        used_neurons = set()
        for conn in self.connections:
            if conn['source_type'] == 1:  # NEURON
                used_neurons.add(conn['source_num'])
            if conn['sink_type'] == 0:  # NEURON
                used_neurons.add(conn['sink_num'])

        self.active_internal_neurons = len(used_neurons)
        self.total_connections = len(self.connections)

    def initialize_action_indices(self):
        """Map action types to their indices in the neural network output"""
        self.action_indices = {
            'move_x': Action.MOVE_X.value,
            'move_y': Action.MOVE_Y.value,
            'move_forward': Action.MOVE_FORWARD.value,
            'move_reverse': Action.MOVE_REVERSE.value,
            'move_left': Action.MOVE_LEFT.value,
            'move_right': Action.MOVE_RIGHT.value,
            'move_rl': Action.MOVE_RL.value,
            'move_random': Action.MOVE_RANDOM.value,
            'move_east': Action.MOVE_EAST.value,
            'move_west': Action.MOVE_WEST.value,
            'move_north': Action.MOVE_NORTH.value,
            'move_south': Action.MOVE_SOUTH.value,
            'set_oscillator_period': Action.SET_OSCILLATOR_PERIOD.value,
            'set_longprobe_dist': Action.SET_LONGPROBE_DIST.value,
            'set_responsiveness': Action.SET_RESPONSIVENESS.value,
            'emit_signal0': Action.EMIT_SIGNAL0.value,
            'kill_forward': Action.KILL_FORWARD.value
        }

    def _post_process_actions(self, action_values, creature_state=None):
        """
        Process actions after neural net processing, particularly for movement coordination
        and special actions like oscillator period setting

        Args:
            action_values: Raw action output values from neural network
            creature_state: Creature's current state information

        Returns:
            Dictionary of state updates to apply to the creature
        """
        state_updates = {}

        # Extract responsiveness if available (affects all actions)
        responsiveness = 1.0
        if Action.SET_RESPONSIVENESS.value < len(action_values):
            responsiveness_raw = action_values[Action.SET_RESPONSIVENESS.value]
            responsiveness = max(0.0, min(1.0, responsiveness_raw))
            state_updates['responsiveness'] = responsiveness

        # Apply the responsiveness curve from C++ implementation
        adjusted_responsiveness = self._apply_responsiveness_curve(responsiveness)

        # Process oscillator period setting
        if Action.SET_OSCILLATOR_PERIOD.value < len(action_values):
            period_val = action_values[Action.SET_OSCILLATOR_PERIOD.value]
            # Convert to 0.0..1.0 range
            period_01 = (math.tanh(period_val) + 1.0) / 2.0
            # Calculate new period using the exact formula from C++
            new_period = 1 + int(1.5 + math.exp(7.0 * period_01))
            # Clamp to reasonable range
            new_period = max(2, min(new_period, 2048))
            state_updates['oscPeriod'] = new_period

        # Process long probe distance
        if Action.SET_LONGPROBE_DIST.value < len(action_values):
            max_probe_dist = 32  # Maximum probe distance
            level = action_values[Action.SET_LONGPROBE_DIST.value]
            level = (math.tanh(level) + 1.0) / 2.0  # Convert to 0.0..1.0
            probe_dist = 1 + int(level * max_probe_dist)
            state_updates['longprobe_dist'] = probe_dist

        # Process signal emission with the same threshold and probability as C++
        if Action.EMIT_SIGNAL0.value < len(action_values):
            emit_threshold = 0.5  # Same as C++ implementation
            level = action_values[Action.EMIT_SIGNAL0.value]
            level *= adjusted_responsiveness

            # Signal emission is probabilistic based on the level
            if level > emit_threshold and random.random() < level:
                state_updates['emit_signal0'] = True

        # Process kill forward action with probabilistic execution
        if Action.KILL_FORWARD.value < len(action_values):
            kill_threshold = 0.5  # Same as C++
            level = action_values[Action.KILL_FORWARD.value]
            level *= adjusted_responsiveness

            if level > kill_threshold and random.random() < (level - self.ACTION_MIN) / (
                    self.ACTION_MAX - self.ACTION_MIN):
                state_updates['attempt_kill'] = True

        # Add other special action processing as needed

        # Process kill forward action with probabilistic execution
        # Match C++ threshold and probability calculation
        if Action.KILL_FORWARD.value < len(action_values):
            kill_threshold = 0.5  # Same as C++
            level = action_values[Action.KILL_FORWARD.value]
            level *= adjusted_responsiveness

            # C++ uses: prob2bool((level - ACTION_MIN) / ACTION_RANGE)
            # Assuming ACTION_MIN=0.0, ACTION_MAX=1.0, ACTION_RANGE=1.0
            if level > kill_threshold and random.random() < level:
                state_updates['attempt_kill'] = True

        # Add other special action processing as needed

        return state_updates

    def _apply_responsiveness_curve(self, r):
        """
        Apply the responsiveness curve from C++ implementation
        that reduces activity level (makes creatures less jittery)
        """
        k = self.params.get('responsiveness_curve_k_factor', 2) # Use Python parameter name
        # Direct translation of the C++ implementation
        return math.pow((r - 2.0), -2.0 * k) - math.pow(2.0, -2.0 * k) * (1.0 - r)

    def process_movement(self, action_values, creature_state):
        """
        Process all movement-related actions and return a movement vector
        
        Args:
            action_values: Neural network output values
            creature_state: Dictionary with creature state (direction, responsiveness)
            
        Returns:
            dict: Movement information including:
                - 'vector': (move_x, move_y) continuous movement vector
                - 'discrete': (dx, dy) discrete grid movement
                - 'direction_update': New direction if movement occurs
        """
        # Extract creature state
        direction = creature_state.get('direction', (0, 0))
        responsiveness = creature_state.get('responsiveness', 1.0)
        
        # Calculate continuous movement vector from all movement neurons
        move_x, move_y = self._calculate_movement_vector(action_values, direction)
        
        # Apply responsiveness
        adjusted_responsiveness = self._apply_responsiveness_curve(responsiveness)
        move_x_final = move_x * adjusted_responsiveness
        move_y_final = move_y * adjusted_responsiveness
        
        # Convert to discrete grid movement
        dx, dy = self._convert_to_discrete_movement(move_x_final, move_y_final)
        
        # print(f"DEBUG NN Move: raw=({move_x:.3f}, {move_y:.3f}), resp={responsiveness:.3f}, adj_resp={adjusted_responsiveness:.3f}, final=({move_x_final:.3f}, {move_y_final:.3f}), discrete=({dx}, {dy})") # DEBUG
        
        # Calculate new direction if movement occurs
        new_direction = None
        if dx != 0 or dy != 0:
            new_direction = self._calculate_new_direction(dx, dy)
        
        return {
            'vector': (move_x, move_y),
            'discrete': (dx, dy),
            'direction_update': new_direction
        }

    def _calculate_movement_vector(self, action_values, direction):
        """Calculate continuous movement vector from all movement neurons"""
        move_x = 0.0
        move_y = 0.0
        
        # Basic X/Y movement
        if Action.MOVE_X.value < len(action_values):
            move_x += (action_values[Action.MOVE_X.value] * 2 - 1)  # Convert from [0,1] to [-1,1]
        
        if Action.MOVE_Y.value < len(action_values):
            move_y += (action_values[Action.MOVE_Y.value] * 2 - 1)  # Convert from [0,1] to [-1,1]
        
        # Cardinal directions
        if Action.MOVE_EAST.value < len(action_values):
            move_x += action_values[Action.MOVE_EAST.value]
        if Action.MOVE_WEST.value < len(action_values):
            move_x -= action_values[Action.MOVE_WEST.value]
        if Action.MOVE_NORTH.value < len(action_values):
            move_y += action_values[Action.MOVE_NORTH.value]
        if Action.MOVE_SOUTH.value < len(action_values):
            move_y -= action_values[Action.MOVE_SOUTH.value]
        
        # Directional movement
        if Action.MOVE_FORWARD.value < len(action_values):
            move_x += direction[0] * action_values[Action.MOVE_FORWARD.value]
            move_y += direction[1] * action_values[Action.MOVE_FORWARD.value]
        
        if Action.MOVE_REVERSE.value < len(action_values):
            move_x -= direction[0] * action_values[Action.MOVE_REVERSE.value]
            move_y -= direction[1] * action_values[Action.MOVE_REVERSE.value]
        
        # Left/right movement (perpendicular to current direction)
        if Action.MOVE_LEFT.value < len(action_values):
            move_x -= direction[1] * action_values[Action.MOVE_LEFT.value]
            move_y += direction[0] * action_values[Action.MOVE_LEFT.value]
        
        if Action.MOVE_RIGHT.value < len(action_values):
            move_x += direction[1] * action_values[Action.MOVE_RIGHT.value]
            move_y -= direction[0] * action_values[Action.MOVE_RIGHT.value]
        
        # Random movement
        if Action.MOVE_RANDOM.value < len(action_values):
            angle = random.uniform(0, 2 * math.pi)
            move_x += math.cos(angle) * action_values[Action.MOVE_RANDOM.value]
            move_y += math.sin(angle) * action_values[Action.MOVE_RANDOM.value]
        
        return move_x, move_y

    def _convert_to_discrete_movement(self, move_x, move_y):
        """Convert continuous movement to discrete grid movement"""
        dx, dy = 0, 0
        if random.random() < abs(move_x):
            dx = 1 if move_x > 0 else -1
        if random.random() < abs(move_y):
            dy = 1 if move_y > 0 else -1
        return dx, dy

    def _calculate_new_direction(self, dx, dy):
        """Calculate new direction vector from discrete movement"""
        if dx == 0 and dy == 0:
            return None
            
        # Normalize direction vector
        length = math.sqrt(dx * dx + dy * dy)
        return (dx / length, dy / length)
        
    def process_movement_for_creature(self, action_values, creature):
        """
        Process complex movement mechanics using the creature's current state
        
        Args:
            action_values: Neural network output values
            creature: The creature whose movements are being processed
            
        Returns:
            Tuple of (dx, dy) representing discretized movement
        """
        # Create creature state dictionary
        creature_state = {
            'direction': creature.direction,
            'responsiveness': creature.responsiveness
        }
        
        # Use the new consolidated movement processing
        movement_info = self.process_movement(action_values, creature_state)
        
        # Return just the discrete movement for backward compatibility
        return movement_info['discrete']

    def process_non_movement_actions(self, action_values, previous_state=None):
        """
        Process non-movement actions like oscillator period settings,
        probe distance, and signal emission

        Args:
            action_values: The raw action neuron output values
            previous_state: Optional dictionary with prior creature state

        Returns:
            Dictionary with updated creature state values
        """
        state_updates = {}

        # Extract responsiveness from action values if present
        responsiveness = 1.0
        if Action.SET_RESPONSIVENESS.value < len(action_values):
            responsiveness = action_values[Action.SET_RESPONSIVENESS.value]
            state_updates['responsiveness'] = responsiveness

        # Process oscillator period setting
        if Action.SET_OSCILLATOR_PERIOD.value < len(action_values):
            periodf = action_values[Action.SET_OSCILLATOR_PERIOD.value]
            periodf01 = (math.tanh(periodf) + 1.0) / 2.0  # Convert to 0.0..1.0
            newPeriod = 1 + int(1.5 + math.exp(7.0 * periodf01))
            state_updates['oscPeriod'] = max(2, min(newPeriod, 2048))  # Range limit

        # Process long probe distance setting
        if Action.SET_LONGPROBE_DIST.value < len(action_values):
            max_probe_dist = 32  # Maximum probe distance
            level = action_values[Action.SET_LONGPROBE_DIST.value]
            probe_dist = 1 + int(level * max_probe_dist)
            state_updates['longProbeDistance'] = probe_dist

        # Signal emission processing logic
        if Action.EMIT_SIGNAL0.value < len(action_values):
            emit_threshold = 0.5  # Threshold for emission
            level = action_values[Action.EMIT_SIGNAL0.value]
            level *= responsiveness
            if level > emit_threshold:
                # Store signal emission flag - actual emission happens in creature
                state_updates['emit_signal0'] = level

        # Kill forward logic with similar threshold
        if Action.KILL_FORWARD.value < len(action_values):
            kill_threshold = 0.5
            level = action_values[Action.KILL_FORWARD.value]
            level *= responsiveness
            if level > kill_threshold:
                state_updates['attempt_kill'] = level

        return state_updates
