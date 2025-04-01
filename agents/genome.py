import random
import logging
import numpy as np

from core.types import Action, Sensor
from utils.random_generator import random_generator
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='genome_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)


class Gene:
    def __init__(self, hex_value=None, params=None):
        """
        Initialize a gene with either a provided hex value or a random one.

        Args:
            hex_value (str, optional): Hexadecimal value representing the gene
            params (dict, optional): Simulation parameters (for weight_divisor)
        """
        self.params = params if params else {} # Store params
        try:
            # Generate or validate hex value
            if hex_value is None:
                self.hex_value = ''.join(
                    format(random_generator.random_uint(0, 15), 'x')
                    for _ in range(8)
                )
            else:
                self.hex_value = str(hex_value)

            # Validate and normalize hex value
            if not all(c in '0123456789abcdefABCDEF' for c in self.hex_value):
                self.hex_value = ''.join(random.choice('0123456789abcdef') for _ in range(8))

            # Ensure exactly 8 characters
            self.hex_value = self.hex_value.zfill(8)[:8].lower()

            # Parse gene fields
            self._parse_hex_value()

        except Exception as e:
            logger.error(f"Error initializing Gene: {e}", exc_info=True)
            raise

    def _parse_hex_value(self):
        """
        Parse all gene fields from the hex value with robust error handling.
        """
        try:
            # Convert hex to binary, ensuring 32-bit representation
            binary = bin(int(self.hex_value, 16))[2:].zfill(32)

            # Parse individual fields
            self.source_type = int(binary[0], 2)  # 0=SENSOR, 1=NEURON
            self.source_id = int(binary[1:8], 2)
            self.sink_type = int(binary[8], 2)  # 0=NEURON, 1=ACTION
            self.sink_id = int(binary[9:16], 2)

            # Weight conversion with signed integer handling
            weight_bits = binary[16:]
            # Use weight_divisor from params, default to C++ value if not found
            weight_divisor = self.params.get('weight_divisor', 8192.0)

            if weight_bits[0] == '1':  # Negative number in two's complement
                self.weight = (int(weight_bits, 2) - 65536) / weight_divisor
            else:
                self.weight = int(weight_bits, 2) / weight_divisor

            logger.debug(f"Parsed gene: "
                         f"source_type={self.source_type}, "
                         f"source_id={self.source_id}, "
                         f"sink_type={self.sink_type}, "
                         f"sink_id={self.sink_id}, "
                         f"weight={self.weight:.4f}")

        except ValueError as ve:
            logger.error(f"Invalid hex value parsing: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing hex value: {e}", exc_info=True)
            raise

    def update_hex_from_fields(self):
        """
        Update hex value from the gene's fields with comprehensive logging.
        """
        try:
            # Format individual fields
            source_type_bits = format(self.source_type & 1, '01b')
            source_id_bits = format(self.source_id & 0x7F, '07b')
            sink_type_bits = format(self.sink_type & 1, '01b')
            sink_id_bits = format(self.sink_id & 0x7F, '07b')

            # Convert weight to 16-bit signed integer
            weight_int = int(self.weight * 6000)
            if weight_int < 0:
                weight_int = 65536 + weight_int  # Convert to two's complement
            weight_bits = format(weight_int & 0xFFFF, '016b')

            # Combine all bits
            binary = source_type_bits + source_id_bits + sink_type_bits + sink_id_bits + weight_bits

            # Convert to hex
            self.hex_value = format(int(binary, 2), '08x')

            logger.debug(f"Updated hex value: {self.hex_value}")
            return self.hex_value

        except Exception as e:
            logger.error(f"Error updating hex value from fields: {e}", exc_info=True)
            raise


class Genome:
    def __init__(self, genes=None, length=32, params=None):
        """
        Initialize a genome with enhanced diversity and balanced movement.

        Args:
            genes (list, optional): Predefined list of genes
            length (int, optional): Number of genes to generate if not provided
            params (dict, optional): Simulation parameters
        """
        try:
            # Initialize parameters
            self.params = params or {}

            # Set genome parameters with logging
            self.genome_initial_length_min = self.params.get('genomeInitialLengthMin', 24)
            self.genome_initial_length_max = self.params.get('genomeInitialLengthMax', 48)
            self.genome_max_length = self.params.get('genomeMaxLength', 300)
            self.genes_insertion_deletion_rate = self.params.get('geneInsertionDeletionRate', 0.0)
            self.deletion_ratio = self.params.get('deletionRatio', 0.5)

            # Create or use provided genes
            if genes is None:
                # Determine length with randomization
                if self.genome_initial_length_min != self.genome_initial_length_max:
                    length = random.randint(self.genome_initial_length_min, self.genome_initial_length_max)

                # Generate genes, passing params
                self.genes = [Gene(params=self.params) for _ in range(length)]

                # Enhance gene diversity
                self.initialize_with_more_diversity()
                self.ensure_basic_movement_genes()
            else:
                # Use provided genes
                self.genes = genes

        except Exception as e:
            logger.error(f"Genome initialization failed: {e}", exc_info=True)
            print(f"Genome initialization error: {e}")
            # Initialize with empty genes list as fallback
            self.genes = []
            self.params = params or {}
            self.genome_max_length = self.params.get('genomeMaxLength', 300)
            self.genes_insertion_deletion_rate = 0.0
            self.deletion_ratio = 0.5

    def mutate(self, mutation_rate):
        """
        Apply multiple types of mutations to the genome.

        Args:
            mutation_rate (float): Probability of mutation for each gene

        Returns:
            int: Number of mutations applied
        """
        start_gene_count = len(self.genes)
        mutations_applied = 0


        try:
            # First pass: Standard per-gene mutations
            for gene in self.genes:
                if random.random() < mutation_rate:
                    self._apply_point_mutation(gene)
                    mutations_applied += 1

            # Insertion/deletion mutations
            if self.genes_insertion_deletion_rate > 0:
                if random.random() < self.genes_insertion_deletion_rate:
                    # Deletion mutation
                    if random.random() < self.deletion_ratio and len(self.genes) > 1:
                        del_index = random.randint(0, len(self.genes) - 1)
                        del self.genes[del_index]
                        mutations_applied += 1
                        logger.debug(f"Deleted gene at index {del_index}")

                    # Insertion mutation
                    elif len(self.genes) < self.genome_max_length:
                        new_gene = Gene(params=self.params) # Pass params
                        self.genes.append(new_gene)
                        mutations_applied += 1
                        logger.debug("Inserted new random gene")

            # Fourth pass: Occasionally awaken movement genes
            if random.random() < 0.3:
                additional_mutations = self.awaken_movement_genes()
                mutations_applied += additional_mutations


            return mutations_applied

        except Exception as e:
            logger.error(f"Genome mutation failed: {e}", exc_info=True)
            return 0

    def _apply_point_mutation(self, gene):
        """
        Apply a point mutation to a single gene.

        Args:
            gene (Gene): The gene to mutate
        """
        try:
            mutation_type = random_generator.random_float()

            if mutation_type < 0.6:  # 60% chance: Bit flip mutation
                # Convert to binary
                binary = bin(int(gene.hex_value, 16))[2:].zfill(32)
                binary_list = list(binary)

                # Flip a random bit
                bit_pos = random_generator.random_uint(0, 31)
                binary_list[bit_pos] = '1' if binary_list[bit_pos] == '0' else '0'

                # Convert back to hex
                gene.hex_value = format(int(''.join(binary_list), 2), '08x')

                # Update parsed fields
                gene._parse_hex_value()
                logger.debug(f"Performed bit flip mutation at position {bit_pos}")

            elif mutation_type < 0.8:  # 20% chance: Change connection
                # Determine whether to change source or sink
                if random.random() < 0.5:
                    # Change source neuron
                    gene.source_type = 1 if random.random() < 0.5 else 0
                    gene.source_id = random.randint(0, 127)
                else:
                    # Change sink neuron
                    gene.sink_type = 1 if random.random() < 0.5 else 0
                    gene.sink_id = random.randint(0, 127)

                # Update hex representation
                gene.update_hex_from_fields()
                logger.debug("Performed connection type mutation")

            else:  # 20% chance: Significant weight change
                # Apply a completely new weight
                sign = 1 if random.random() < 0.5 else -1
                gene.weight = sign * random.uniform(0.2, 0.8)

                # Update hex representation
                gene.update_hex_from_fields()
                logger.debug(f"Performed weight mutation: new weight = {gene.weight:.4f}")

        except Exception as e:
            logger.error(f"Point mutation failed: {e}", exc_info=True)
            raise

    def crossover(self, other_genome):
        """
        Create a child genome through crossover with another genome.
        This implementation closely follows the C++ biosim4 approach.

        Args:
            other_genome (Genome): The genome to cross with

        Returns:
            Genome: A new genome created by combining parent genomes.
        """
        try:
            # Handle empty genomes
            if not self.genes or not other_genome.genes:
                genes_to_use = self.genes if self.genes else other_genome.genes
                child_genes = [Gene(g.hex_value, params=self.params) for g in genes_to_use]
                return Genome(genes=child_genes, params=self.params)

            # Create a new child genome
            child_genome = None
            
            # Determine which parent has the longer genome
            if len(self.genes) > len(other_genome.genes):
                longer_genome = self
                shorter_genome = other_genome
            else:
                longer_genome = other_genome
                shorter_genome = self
                
            # Start with the longer genome as the base
            child_genes = [Gene(g.hex_value, params=self.params) for g in longer_genome.genes]
            child_genome = Genome(genes=child_genes, params=self.params)
            
            # Overlay a slice of the shorter genome onto the child
            # This matches the C++ implementation's overlayWithSliceOf function
            shorter_len = len(shorter_genome.genes)
            if shorter_len > 0:
                # Choose random start and end points for the slice
                index0 = random_generator.random_uint(0, shorter_len - 1)
                index1 = random_generator.random_uint(0, shorter_len)
                
                # Ensure index0 <= index1
                if index0 > index1:
                    index0, index1 = index1, index0
                
                # Copy the slice from shorter genome to child
                for i in range(index0, min(index1, len(child_genome.genes))):
                    if i < len(shorter_genome.genes):
                        child_genome.genes[i] = Gene(shorter_genome.genes[i].hex_value, params=self.params)
            
            # Adjust length to average of parents (matching C++ implementation)
            target_length = (len(self.genes) + len(other_genome.genes)) // 2
            
            # If average length is not an integer, randomly add 1 half the time
            if (len(self.genes) + len(other_genome.genes)) % 2 == 1 and random_generator.random_float() < 0.5:
                target_length += 1
                
            # Ensure target length is within valid range
            target_length = max(1, min(target_length, self.genome_max_length))
            
            # Adjust genome length to target
            if len(child_genome.genes) > target_length:
                child_genome.genes = child_genome.genes[:target_length]
            elif len(child_genome.genes) < target_length:
                # Add random genes if needed
                for _ in range(target_length - len(child_genome.genes)):
                    child_genome.genes.append(Gene(params=self.params))
            
            return child_genome

        except Exception as e:
            logger.error(f"Genome crossover failed: {e}", exc_info=True)
            # Return a copy of one parent as fallback
            fallback_genes = [Gene(g.hex_value, params=self.params) for g in self.genes]
            return Genome(genes=fallback_genes, params=self.params)

    def ensure_basic_movement_genes(self):
        """
        Ensure the genome has at least one functional connection for basic movement capabilities.
        """
        try:
            # List of essential movement actions
            essential_actions = [
                Action.MOVE_X.value,
                Action.MOVE_Y.value,
                Action.MOVE_FORWARD.value
            ]

            # Get existing actions in the genome
            existing_actions = set()
            for gene in self.genes:
                if gene.sink_type == 1 and abs(gene.weight) > 0.1:  # ACTION with significant weight
                    existing_actions.add(gene.sink_id)

            # Find missing essential actions
            missing_actions = [action for action in essential_actions if action not in existing_actions]

            # Add missing essential action connections if there are genome slots available
            if missing_actions and len(self.genes) < self.genome_max_length:
                for action in missing_actions:
                    # Create a new gene for this action, passing params
                    new_gene = Gene(params=self.params)

                    # Configure the gene for basic movement
                    new_gene.sink_type = 1  # ACTION
                    new_gene.sink_id = action
                    new_gene.source_type = 0  # SENSOR

                    # Connect to a useful sensor for movement
                    if action == Action.MOVE_X.value:
                        new_gene.source_id = Sensor.LOC_X.value
                        new_gene.weight = 0.4 * (-1 if random.random() < 0.5 else 1)
                    elif action == Action.MOVE_Y.value:
                        new_gene.source_id = Sensor.LOC_Y.value
                        new_gene.weight = 0.4 * (-1 if random.random() < 0.5 else 1)
                    elif action == Action.MOVE_FORWARD.value:
                        new_gene.source_id = Sensor.RANDOM.value
                        new_gene.weight = 0.5

                    # Update hex representation
                    new_gene.update_hex_from_fields()

                    # Add to genome
                    self.genes.append(new_gene)
        except Exception as e:
            logger.error(f"Failed to ensure basic movement genes: {e}", exc_info=True)
            raise

    def balance_movement_weights(self):
        """
        Balance movement gene weights to prevent directional bias.

        This method analyzes movement-related genes and adjusts their weights
        to reduce potential directional biases in creature movement.
        """
        try:
            # Identify movement-related genes
            movement_genes = {
                'x_movement': [],  # MOVE_X
                'y_movement': [],  # MOVE_Y
                'cardinal': {  # Directional movement genes
                    'east': [],  # MOVE_EAST
                    'west': [],  # MOVE_WEST
                    'north': [],  # MOVE_NORTH
                    'south': []  # MOVE_SOUTH
                }
            }

            # Categorize movement genes
            for gene in self.genes:
                if gene.sink_type == 1:  # ACTION
                    action_id = gene.sink_id % 128

                    if action_id == 0:  # MOVE_X
                        movement_genes['x_movement'].append(gene)
                    elif action_id == 1:  # MOVE_Y
                        movement_genes['y_movement'].append(gene)
                    elif action_id == 9:  # MOVE_EAST
                        movement_genes['cardinal']['east'].append(gene)
                    elif action_id == 10:  # MOVE_WEST
                        movement_genes['cardinal']['west'].append(gene)
                    elif action_id == 11:  # MOVE_NORTH
                        movement_genes['cardinal']['north'].append(gene)
                    elif action_id == 12:  # MOVE_SOUTH
                        movement_genes['cardinal']['south'].append(gene)

            # Calculate average weights for different movement directions
            def calculate_avg_weight(gene_list):
                return sum(gene.weight for gene in gene_list) / len(gene_list) if gene_list else 0

            # Calculate corrections for different movement dimensions
            east_west_avg = calculate_avg_weight(movement_genes['cardinal']['east']) - \
                            calculate_avg_weight(movement_genes['cardinal']['west'])
            north_south_avg = calculate_avg_weight(movement_genes['cardinal']['north']) - \
                              calculate_avg_weight(movement_genes['cardinal']['south'])
            x_avg = calculate_avg_weight(movement_genes['x_movement'])
            y_avg = calculate_avg_weight(movement_genes['y_movement'])

            # Apply weight corrections
            correction_factor = 0.7  # Moderate correction to prevent overcorrection

            # Correct cardinal direction genes
            for direction, opposite in [
                ('east', 'west'),
                ('west', 'east'),
                ('north', 'south'),
                ('south', 'north')
            ]:
                correction = east_west_avg if direction in ['east', 'west'] else north_south_avg
                for gene in movement_genes['cardinal'][direction]:
                    gene.weight -= correction * correction_factor
                    gene.update_hex_from_fields()

            # Correct X and Y movement genes
            for movement_type, avg in [('x_movement', x_avg), ('y_movement', y_avg)]:
                for gene in movement_genes[movement_type]:
                    gene.weight -= avg * correction_factor
                    gene.update_hex_from_fields()


        except Exception as e:
            logger.error(f"Failed to balance movement gene weights: {e}", exc_info=True)

    def initialize_with_more_diversity(self, movement_bias=0.6, activation_ratio=0.7):
        """
        Initialize the genome with diverse genes focused on movement actions.

        Args:
            movement_bias (float): Probability of creating movement-related connections
            activation_ratio (float): Percentage of genes with significant non-zero weights
        """
        try:

            # Identify movement-related action indices
            movement_actions = [
                Action.MOVE_X.value, Action.MOVE_Y.value,
                Action.MOVE_EAST.value, Action.MOVE_WEST.value,
                Action.MOVE_NORTH.value, Action.MOVE_SOUTH.value,
                Action.MOVE_FORWARD.value, Action.MOVE_REVERSE.value,
                Action.MOVE_LEFT.value, Action.MOVE_RIGHT.value,
                Action.MOVE_RL.value, Action.MOVE_RANDOM.value
            ]

            # Identify useful sensory inputs for movement
            movement_sensors = [
                Sensor.LOC_X.value, Sensor.LOC_Y.value,
                Sensor.BOUNDARY_DIST_X.value, Sensor.BOUNDARY_DIST_Y.value,
                Sensor.BOUNDARY_DIST.value, Sensor.LAST_MOVE_DIR_X.value,
                Sensor.LAST_MOVE_DIR_Y.value, Sensor.LONGPROBE_POP_FWD.value,
                Sensor.LONGPROBE_BAR_FWD.value, Sensor.BARRIER_FWD.value,
                Sensor.BARRIER_LR.value, Sensor.POPULATION_FWD.value
            ]

            # For each gene, decide if it should be a movement gene
            for gene in self.genes:
                if random.random() < movement_bias:
                    # Create a movement-related gene
                    gene.sink_type = 1  # ACTION
                    gene.sink_id = random.choice(movement_actions)

                    # Connect to a useful sensor or internal neuron
                    if random.random() < 0.7:  # 70% connect to sensor
                        gene.source_type = 0  # SENSOR
                        gene.source_id = random.choice(movement_sensors)
                    else:  # 30% connect to internal neuron
                        gene.source_type = 1  # NEURON
                        gene.source_id = random.randint(0, self.params.get('num_internal_neurons', 16) - 1)

                    # Assign meaningful weights
                    if random.random() < activation_ratio:
                        # Generate weight with a distribution that avoids values near zero
                        if random.random() < 0.5:
                            # Positive weight - either weak or strong
                            if random.random() < 0.5:
                                gene.weight = random.uniform(0.1, 0.3)  # Weak positive
                            else:
                                gene.weight = random.uniform(0.3, 0.8)  # Strong positive
                        else:
                            # Negative weight - either weak or strong
                            if random.random() < 0.5:
                                gene.weight = random.uniform(-0.3, -0.1)  # Weak negative
                            else:
                                gene.weight = random.uniform(-0.8, -0.3)  # Strong negative
                    else:
                        # Some connections remain dormant (near zero) initially
                        gene.weight = random.uniform(-0.05, 0.05)
                else:
                    # Non-movement gene - other behaviors like responsiveness, oscillator, signal emission
                    gene.sink_type = random.randint(0, 1)  # Either NEURON or ACTION

                    if gene.sink_type == 1:  # ACTION
                        non_movement_actions = [a for a in range(self.params.get('num_output_neurons', 17))
                                                if a not in movement_actions]
                        gene.sink_id = random.choice(non_movement_actions) if non_movement_actions else 0
                    else:  # NEURON
                        gene.sink_id = random.randint(0, self.params.get('num_internal_neurons', 16) - 1)

                    # Source type and ID
                    gene.source_type = random.randint(0, 1)
                    if gene.source_type == 0:  # SENSOR
                        gene.source_id = random.randint(0, self.params.get('num_sensory_neurons', 24) - 1)
                    else:  # NEURON
                        gene.source_id = random.randint(0, self.params.get('num_internal_neurons', 16) - 1)

                    # Assign weights with lower activation ratio for non-movement genes
                    if random.random() < (activation_ratio * 0.7):
                        gene.weight = random.uniform(-0.5, 0.5)
                    else:
                        gene.weight = random.uniform(-0.05, 0.05)

                # Update hex representation
                gene.update_hex_from_fields()

            # Balance initial movement weights to prevent directional bias
            self.balance_movement_weights()


        except Exception as e:
            logger.error(f"Failed to initialize genome diversity: {e}", exc_info=True)

    def awaken_movement_genes(self, cardinal_bias=0.7):
        """
        Specialized mutation that specifically targets and activates
        genes connected to movement outputs.

        Args:
            cardinal_bias (float): Probability of focusing on cardinal directions vs X/Y

        Returns:
            int: Number of genes activated
        """
        try:

            # Identify genes connected to movement outputs
            movement_genes = {
                'x_movement': [],  # MOVE_X
                'y_movement': [],  # MOVE_Y
                'cardinal': []  # EAST, WEST, NORTH, SOUTH
            }

            for gene in self.genes:
                if gene.sink_type == 1:  # ACTION
                    sink_id = gene.sink_id % 128

                    # Categorize by movement type
                    if sink_id == 0:  # MOVE_X
                        movement_genes['x_movement'].append(gene)
                    elif sink_id == 1:  # MOVE_Y
                        movement_genes['y_movement'].append(gene)
                    elif 9 <= sink_id <= 12:  # EAST, WEST, NORTH, SOUTH
                        movement_genes['cardinal'].append(gene)

            # Choose which category to focus on
            if random.random() < cardinal_bias and movement_genes['cardinal']:
                target_genes = movement_genes['cardinal']
                # Strong activation for cardinal directions
                activation_count = min(len(target_genes), max(2, len(target_genes) // 3))
            else:
                # Combine X and Y movement genes
                target_genes = movement_genes['x_movement'] + movement_genes['y_movement']
                # Moderate activation for X/Y
                activation_count = min(len(target_genes), max(1, len(target_genes) // 5))

            # If no movement genes found, early return
            if not target_genes:
                return 0

            # Activate randomly selected movement genes
            activated = 0
            for _ in range(activation_count):
                # Select random gene from the target category
                gene = random.choice(target_genes)

                # Only activate if weight is currently near zero
                if abs(gene.weight) < 0.1:
                    # Set to a stronger non-zero value
                    sign = 1 if random.random() < 0.5 else -1
                    strength = random.uniform(0.2, 0.6)
                    gene.weight = sign * strength

                    # Update the hex representation
                    gene.update_hex_from_fields()
                    activated += 1

            return activated

        except Exception as e:
            logger.error(f"Failed to awaken movement genes: {e}", exc_info=True)
            return 0

    def hash(self):
        """
        Generate a hash representation of the genome.

        Returns:
            str: A concatenated string of gene hex values
        """
        try:
            genome_hash = ''.join(gene.hex_value for gene in self.genes)
            return genome_hash
        except Exception as e:
            logger.error(f"Failed to generate genome hash: {e}", exc_info=True)
            return ""

    def calculate_similarity(self, other_genome):
        """
        Calculate genetic similarity between two genomes.

        Args:
            other_genome (Genome): The genome to compare against

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            # If genomes have different lengths, use the shorter one for comparison
            min_length = min(len(self.genes), len(other_genome.genes))

            if min_length == 0:
                return 0.0

            # Count exact matches
            matching_genes = 0
            for i in range(min_length):
                if self.genes[i].hex_value == other_genome.genes[i].hex_value:
                    matching_genes += 1

            # Normalize by the length of the shorter genome
            similarity = matching_genes / min_length

            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate genome similarity: {e}", exc_info=True)
            return 0.0

    @classmethod
    def normalize_genome_weights(cls, genome):
        """
        Normalize genome weights to reduce directional bias.

        Args:
            genome (Genome): The genome to normalize
        """
        try:

            # Identify movement-related genes
            movement_genes = {
                'MOVE_X': [], 'MOVE_Y': [],
                'MOVE_EAST': [], 'MOVE_WEST': [],
                'MOVE_NORTH': [], 'MOVE_SOUTH': [],
                'MOVE_FORWARD': [], 'MOVE_RL': []
            }

            # Categorize movement genes
            for gene in genome.genes:
                if gene.sink_type == 1:  # ACTION
                    action_id = gene.sink_id % 128
                    if action_id == 0:  # MOVE_X
                        movement_genes['MOVE_X'].append(gene)
                    elif action_id == 1:  # MOVE_Y
                        movement_genes['MOVE_Y'].append(gene)
                    elif action_id == 9:  # MOVE_EAST
                        movement_genes['MOVE_EAST'].append(gene)
                    elif action_id == 10:  # MOVE_WEST
                        movement_genes['MOVE_WEST'].append(gene)
                    elif action_id == 11:  # MOVE_NORTH
                        movement_genes['MOVE_NORTH'].append(gene)
                    elif action_id == 12:  # MOVE_SOUTH
                        movement_genes['MOVE_SOUTH'].append(gene)
                    elif action_id == 2:  # MOVE_FORWARD
                        movement_genes['MOVE_FORWARD'].append(gene)
                    elif action_id == 3:  # MOVE_RL
                        movement_genes['MOVE_RL'].append(gene)

            # Calculate average weights for different movement directions
            east_avg = sum(g.weight for g in movement_genes['MOVE_EAST']) / len(movement_genes['MOVE_EAST']) if \
            movement_genes['MOVE_EAST'] else 0
            west_avg = sum(g.weight for g in movement_genes['MOVE_WEST']) / len(movement_genes['MOVE_WEST']) if \
            movement_genes['MOVE_WEST'] else 0
            north_avg = sum(g.weight for g in movement_genes['MOVE_NORTH']) / len(
                movement_genes['MOVE_NORTH']) if movement_genes['MOVE_NORTH'] else 0
            south_avg = sum(g.weight for g in movement_genes['MOVE_SOUTH']) / len(
                movement_genes['MOVE_SOUTH']) if movement_genes['MOVE_SOUTH'] else 0
            x_avg = sum(g.weight for g in movement_genes['MOVE_X']) / len(movement_genes['MOVE_X']) if \
            movement_genes['MOVE_X'] else 0
            y_avg = sum(g.weight for g in movement_genes['MOVE_Y']) / len(movement_genes['MOVE_Y']) if \
            movement_genes['MOVE_Y'] else 0

            # Calculate corrections
            east_west_correction = (east_avg - west_avg) * 0.7
            north_south_correction = (north_avg - south_avg) * 0.7
            x_correction = x_avg * 0.7
            y_correction = y_avg * 0.7

            # Apply corrections
            for gene_list, correction in [
                (movement_genes['MOVE_EAST'], -east_west_correction),
                (movement_genes['MOVE_WEST'], east_west_correction),
                (movement_genes['MOVE_NORTH'], -north_south_correction),
                (movement_genes['MOVE_SOUTH'], north_south_correction),
                (movement_genes['MOVE_X'], -x_correction),
                (movement_genes['MOVE_Y'], -y_correction)
            ]:
                for gene in gene_list:
                    gene.weight += correction
                    gene.update_hex_from_fields()


        except Exception as e:
            logger.error(f"Failed to normalize genome weights: {e}", exc_info=True)

    # Genome Validation and Diagnostics
    def validate_genome(self):
        """
        Perform comprehensive validation of the genome.

        Returns:
            dict: Validation results with various genome metrics
        """
        try:

            # Basic structure validation
            validation_results = {
                'is_valid': True,
                'gene_count': len(self.genes),
                'total_weight': 0.0,
                'movement_gene_count': 0,
                'action_gene_distribution': {},
                'issues': []
            }

            # Check maximum genome length
            if len(self.genes) > self.genome_max_length:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Genome exceeds maximum length of {self.genome_max_length}")

            # Analyze genes
            for gene in self.genes:
                # Accumulate total weight
                validation_results['total_weight'] += abs(gene.weight)

                # Count movement-related genes
                if gene.sink_type == 1:  # ACTION
                    action_id = gene.sink_id % 128
                    validation_results['action_gene_distribution'][action_id] = \
                        validation_results['action_gene_distribution'].get(action_id, 0) + 1

                    # Identify movement genes
                    if 0 <= action_id <= 15:  # Adjust range as needed
                        validation_results['movement_gene_count'] += 1

                # Validate individual gene fields
                if gene.source_type not in [0, 1] or gene.sink_type not in [0, 1]:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(f"Invalid source/sink type in gene {gene.hex_value}")

            # Calculate weight distribution metrics
            validation_results['average_weight'] = (validation_results['total_weight'] /
                                                    max(1, len(self.genes)))

            return validation_results

        except Exception as e:
            logger.error(f"Genome validation failed: {e}", exc_info=True)
            return {
                'is_valid': False,
                'error': str(e)
            }

    def compare_genomes(genome1, genome2):
        """
        Perform a comprehensive comparison between two genomes.

        Args:
            genome1 (Genome): First genome to compare
            genome2 (Genome): Second genome to compare

        Returns:
            dict: Detailed comparison results
        """
        try:

            comparison_results = {
                'genetic_similarity': genome1.calculate_similarity(genome2),
                'length_difference': abs(len(genome1.genes) - len(genome2.genes)),
                'gene_by_gene_comparison': [],
                'weight_correlation': None
            }

            # Gene-by-gene comparison
            min_length = min(len(genome1.genes), len(genome2.genes))
            for i in range(min_length):
                gene1, gene2 = genome1.genes[i], genome2.genes[i]

                gene_comparison = {
                    'index': i,
                    'hex_match': gene1.hex_value == gene2.hex_value,
                    'source_type_match': gene1.source_type == gene2.source_type,
                    'sink_type_match': gene1.sink_type == gene2.sink_type,
                    'weight_difference': abs(gene1.weight - gene2.weight)
                }

                comparison_results['gene_by_gene_comparison'].append(gene_comparison)

            # Calculate weight correlation (if possible)
            try:
                weights1 = [gene.weight for gene in genome1.genes[:min_length]]
                weights2 = [gene.weight for gene in genome2.genes[:min_length]]
                correlation = np.corrcoef(weights1, weights2)[0, 1]
                comparison_results['weight_correlation'] = correlation
            except Exception:
                # Correlation calculation might fail for various reasons
                comparison_results['weight_correlation'] = None

            return comparison_results

        except Exception as e:
            logger.error(f"Genome comparison failed: {e}", exc_info=True)
            return {
                'error': str(e)
            }

    def generate_genome_report(genome):
        """
        Generate a comprehensive report about a genome.

        Args:
            genome (Genome): Genome to analyze

        Returns:
            dict: Detailed genome report
        """
        try:

            # Validate the genome first
            validation_results = genome.validate_genome()

            report = {
                'validation': validation_results,
                'gene_statistics': {
                    'total_genes': len(genome.genes),
                    'movement_genes': validation_results.get('movement_gene_count', 0),
                    'action_gene_distribution': validation_results.get('action_gene_distribution', {})
                },
                'weight_analysis': {
                    'total_weight': sum(abs(gene.weight) for gene in genome.genes),
                    'average_weight': validation_results.get('average_weight', 0),
                    'weight_range': {
                        'min': min((gene.weight for gene in genome.genes), default=0),
                        'max': max((gene.weight for gene in genome.genes), default=0)
                    }
                },
                'hex_value_analysis': {
                    'unique_hex_values': len(set(gene.hex_value for gene in genome.genes))
                }
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate genome report: {e}", exc_info=True)
            return {
                'error': str(e)
            }

    # Export functions for global use
    __all__ = [
        'Gene',
        'Genome',
        'validate_genome',
        'compare_genomes',
        'generate_genome_report'
    ]
